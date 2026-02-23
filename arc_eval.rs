// ARC-AGI-2 Evaluation Binary for RustyWorm
// Evaluates the 8-layer multiplicative integration system against ARC-AGI-2 tasks
//
// Usage:
//   cargo run --features layers --bin arc_eval -- [OPTIONS]
//
// Options:
//   --data-dir <path>    Path to ARC data directory (default: arc-data/data)
//   --split <split>      training | evaluation (default: training)
//   --task <id>          Run a single task by ID (e.g. 007bbfb7)
//   --limit <n>          Max tasks to evaluate (default: all)
//   --verbose            Print detailed per-task output
//   --trials <n>         Max trials per test input (default: 2, per ARC-AGI-2 rules)

#[cfg(feature = "layers")]
mod arc {
    use rustyworm::mimicry::layers::bridges::BridgeBuilder;
    use rustyworm::mimicry::layers::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::Instant;

    // ═══════════════════════════════════════════════════════════════════
    // § 1  DATA MODEL
    // ═══════════════════════════════════════════════════════════════════

    pub type Grid = Vec<Vec<u8>>;

    #[derive(Debug, Clone, Deserialize)]
    pub struct ArcPair {
        pub input: Grid,
        pub output: Grid,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ArcTask {
        pub train: Vec<ArcPair>,
        pub test: Vec<ArcPair>,
    }

    #[derive(Debug, Clone)]
    pub struct TaskResult {
        pub task_id: String,
        pub correct: usize,
        pub total: usize,
        pub trials_used: Vec<usize>,
        pub confidence: f32,
        pub time_ms: f64,
    }

    #[derive(Debug, Clone)]
    pub struct EvalSummary {
        pub total_tasks: usize,
        pub tasks_solved: usize,
        pub total_test_inputs: usize,
        pub test_inputs_solved: usize,
        pub avg_confidence: f32,
        pub total_time_ms: f64,
        pub per_category: HashMap<String, (usize, usize)>,
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 2  DATA LOADER
    // ═══════════════════════════════════════════════════════════════════

    pub fn load_task(path: &Path) -> Result<ArcTask, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    pub fn load_tasks(data_dir: &Path, split: &str) -> Result<Vec<(String, ArcTask)>, String> {
        let dir = data_dir.join(split);
        if !dir.exists() {
            return Err(format!("Directory not found: {}", dir.display()));
        }
        let mut tasks = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read dir: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let id = path.file_stem().unwrap().to_string_lossy().to_string();
            let task = load_task(&path)?;
            tasks.push((id, task));
        }
        Ok(tasks)
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 3  GRID ANALYSIS — Feature Extraction
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, Clone)]
    pub struct GridFeatures {
        pub height: usize,
        pub width: usize,
        pub colors: Vec<u8>,
        pub color_counts: HashMap<u8, usize>,
        pub background_color: u8,
        pub num_nonbg_cells: usize,
        pub objects: Vec<Object>,
        pub symmetry: Symmetry,
        pub has_repeating_pattern: bool,
        pub grid_ratio: (usize, usize), // simplest ratio h:w
    }

    #[derive(Debug, Clone)]
    pub struct Object {
        pub color: u8,
        pub cells: Vec<(usize, usize)>,
        pub bbox: (usize, usize, usize, usize), // min_r, min_c, max_r, max_c
        pub area: usize,
    }

    #[derive(Debug, Clone, Default)]
    pub struct Symmetry {
        pub horizontal: bool,
        pub vertical: bool,
        pub diagonal: bool,
        pub rotational_90: bool,
    }

    pub fn analyze_grid(grid: &Grid) -> GridFeatures {
        let height = grid.len();
        let width = if height > 0 { grid[0].len() } else { 0 };

        // Color counts
        let mut color_counts: HashMap<u8, usize> = HashMap::new();
        for row in grid {
            for &c in row {
                *color_counts.entry(c).or_insert(0) += 1;
            }
        }
        let mut colors: Vec<u8> = color_counts.keys().copied().collect();
        colors.sort();

        // Background = most common color
        let background_color = *color_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(c, _)| c)
            .unwrap_or(&0);

        let num_nonbg_cells: usize = color_counts
            .iter()
            .filter(|(&c, _)| c != background_color)
            .map(|(_, &count)| count)
            .sum();

        // Objects via flood fill
        let objects = extract_objects(grid, background_color);

        // Symmetry
        let symmetry = check_symmetry(grid);

        // Repeating pattern check
        let has_repeating_pattern = check_repeating_pattern(grid);

        let g = gcd(height.max(1), width.max(1));
        let grid_ratio = (height / g.max(1), width / g.max(1));

        GridFeatures {
            height,
            width,
            colors,
            color_counts,
            background_color,
            num_nonbg_cells,
            objects,
            symmetry,
            has_repeating_pattern,
            grid_ratio,
        }
    }

    fn extract_objects(grid: &Grid, bg: u8) -> Vec<Object> {
        let h = grid.len();
        let w = if h > 0 { grid[0].len() } else { return vec![] };
        let mut visited = vec![vec![false; w]; h];
        let mut objects = Vec::new();

        for r in 0..h {
            for c in 0..w {
                if !visited[r][c] && grid[r][c] != bg {
                    let color = grid[r][c];
                    let mut cells = Vec::new();
                    let mut stack = vec![(r, c)];
                    visited[r][c] = true;
                    while let Some((cr, cc)) = stack.pop() {
                        cells.push((cr, cc));
                        for (dr, dc) in &[(0i32, 1), (0, -1), (1, 0), (-1, 0)] {
                            let nr = cr as i32 + dr;
                            let nc = cc as i32 + dc;
                            if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                                let (nr, nc) = (nr as usize, nc as usize);
                                if !visited[nr][nc] && grid[nr][nc] == color {
                                    visited[nr][nc] = true;
                                    stack.push((nr, nc));
                                }
                            }
                        }
                    }
                    let min_r = cells.iter().map(|&(r, _)| r).min().unwrap();
                    let max_r = cells.iter().map(|&(r, _)| r).max().unwrap();
                    let min_c = cells.iter().map(|&(_, c)| c).min().unwrap();
                    let max_c = cells.iter().map(|&(_, c)| c).max().unwrap();
                    let area = cells.len();
                    objects.push(Object {
                        color,
                        cells,
                        bbox: (min_r, min_c, max_r, max_c),
                        area,
                    });
                }
            }
        }
        objects
    }

    fn check_symmetry(grid: &Grid) -> Symmetry {
        let h = grid.len();
        let w = if h > 0 {
            grid[0].len()
        } else {
            return Symmetry::default();
        };

        // Horizontal (top-bottom mirror)
        let horizontal = (0..h / 2).all(|r| grid[r] == grid[h - 1 - r]);

        // Vertical (left-right mirror)
        let vertical = (0..h).all(|r| (0..w / 2).all(|c| grid[r][c] == grid[r][w - 1 - c]));

        // Diagonal (transpose == self, only for square)
        let diagonal = if h == w {
            (0..h).all(|r| (0..w).all(|c| grid[r][c] == grid[c][r]))
        } else {
            false
        };

        // 90° rotational
        let rotational_90 = if h == w {
            (0..h).all(|r| (0..w).all(|c| grid[r][c] == grid[c][h - 1 - r]))
        } else {
            false
        };

        Symmetry {
            horizontal,
            vertical,
            diagonal,
            rotational_90,
        }
    }

    fn check_repeating_pattern(grid: &Grid) -> bool {
        let h = grid.len();
        let w = if h > 0 { grid[0].len() } else { return false };

        // Check if the grid is made of tiled copies of a smaller pattern
        for ph in 1..=h / 2 {
            if h % ph != 0 {
                continue;
            }
            for pw in 1..=w / 2 {
                if w % pw != 0 {
                    continue;
                }
                let tile: Vec<Vec<u8>> = (0..ph).map(|r| grid[r][..pw].to_vec()).collect();
                let matches = (0..h).all(|r| (0..w).all(|c| grid[r][c] == tile[r % ph][c % pw]));
                if matches {
                    return true;
                }
            }
        }
        false
    }

    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 4  TRANSFORMATION DETECTION
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, Clone, PartialEq)]
    pub enum Transform {
        Identity,
        ColorMap(HashMap<u8, u8>),
        Scale {
            factor_h: usize,
            factor_w: usize,
        },
        Crop(usize, usize, usize, usize),
        Tile {
            repeat_h: usize,
            repeat_w: usize,
        },
        FlipHorizontal,
        FlipVertical,
        Rotate90,
        Rotate180,
        Rotate270,
        Transpose,
        FillColor {
            from: u8,
            to: u8,
        },
        FillEnclosed {
            boundary_color: u8,
            fill_color: u8,
        },
        Gravity {
            direction: Direction,
        },
        PatternTile, // tile input based on non-bg cell positions
        ConditionalColorSwap {
            condition_color: u8,
            swap: HashMap<u8, u8>,
        },
        OutputSizeChange {
            height: usize,
            width: usize,
        },
        MostCommonObjectColor,
        CopyInputToOutput,
        OutlineObjects {
            outline_color: u8,
        },
        CropToBoundingBox,
        SwapColors {
            a: u8,
            b: u8,
        },
        RemoveSmallObjects {
            min_size: usize,
        },
        ExtractLargestObject,
        HollowObjects {
            hollow_color: u8,
        },
        RecolorBySize {
            color_order: Vec<u8>,
        },
        ConnectDots {
            line_color: u8,
        },
        HalfGrid {
            half: GridHalf,
        },
        Custom(String),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum GridHalf {
        Top,
        Bottom,
        Left,
        Right,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Direction {
        Up,
        Down,
        Left,
        Right,
    }

    #[derive(Debug, Clone)]
    pub struct TransformCandidate {
        pub transform: Transform,
        pub confidence: f32,
        pub correct_on_train: usize,
        pub total_train: usize,
    }

    pub fn detect_transforms(task: &ArcTask) -> Vec<TransformCandidate> {
        let mut candidates = Vec::new();

        // Test each transform type against all training pairs
        let transforms_to_try = generate_transform_hypotheses(task);

        for transform in transforms_to_try {
            let mut correct = 0;
            let total = task.train.len();
            for pair in &task.train {
                if let Some(output) = apply_transform(&pair.input, &transform) {
                    if output == pair.output {
                        correct += 1;
                    }
                }
            }
            if correct > 0 {
                let confidence = correct as f32 / total as f32;
                candidates.push(TransformCandidate {
                    transform,
                    confidence,
                    correct_on_train: correct,
                    total_train: total,
                });
            }
        }

        // Sort by confidence descending, then by simplicity (earlier in list)
        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap()
                .then(b.correct_on_train.cmp(&a.correct_on_train))
        });

        candidates
    }

    fn generate_transform_hypotheses(task: &ArcTask) -> Vec<Transform> {
        let mut hypotheses = Vec::new();

        // Always try identity
        hypotheses.push(Transform::Identity);
        hypotheses.push(Transform::MostCommonObjectColor);

        // Geometric transforms
        hypotheses.push(Transform::FlipHorizontal);
        hypotheses.push(Transform::FlipVertical);
        hypotheses.push(Transform::Rotate90);
        hypotheses.push(Transform::Rotate180);
        hypotheses.push(Transform::Rotate270);
        hypotheses.push(Transform::Transpose);

        // Analyze first training pair for size relationships
        if let Some(pair) = task.train.first() {
            let ih = pair.input.len();
            let iw = if ih > 0 { pair.input[0].len() } else { 0 };
            let oh = pair.output.len();
            let ow = if oh > 0 { pair.output[0].len() } else { 0 };

            // Scale
            if oh > ih && ow > iw && oh % ih == 0 && ow % iw == 0 {
                hypotheses.push(Transform::Scale {
                    factor_h: oh / ih,
                    factor_w: ow / iw,
                });
            }

            // Tiling
            if oh > ih && ow > iw && oh % ih == 0 && ow % iw == 0 {
                hypotheses.push(Transform::Tile {
                    repeat_h: oh / ih,
                    repeat_w: ow / iw,
                });
            }
            // Pattern-based tiling
            if oh > ih || ow > iw {
                hypotheses.push(Transform::PatternTile);
            }

            // Output size change
            if ih != oh || iw != ow {
                hypotheses.push(Transform::OutputSizeChange {
                    height: oh,
                    width: ow,
                });
            }
        }

        // Color mappings from training pairs
        if let Some(color_map) = infer_color_map(task) {
            hypotheses.push(Transform::ColorMap(color_map));
        }

        // Fill transforms
        for pair in &task.train {
            let in_feats = analyze_grid(&pair.input);
            let out_feats = analyze_grid(&pair.output);

            // FillColor: check if one color was uniformly replaced
            for &fc in &in_feats.colors {
                for &tc in &out_feats.colors {
                    if fc != tc {
                        hypotheses.push(Transform::FillColor { from: fc, to: tc });
                    }
                }
            }

            // FillEnclosed
            for &bc in &in_feats.colors {
                for &fc in &out_feats.colors {
                    if bc != fc && bc != in_feats.background_color {
                        hypotheses.push(Transform::FillEnclosed {
                            boundary_color: bc,
                            fill_color: fc,
                        });
                    }
                }
            }
        }

        // Gravity directions
        hypotheses.push(Transform::Gravity {
            direction: Direction::Down,
        });
        hypotheses.push(Transform::Gravity {
            direction: Direction::Up,
        });
        hypotheses.push(Transform::Gravity {
            direction: Direction::Left,
        });
        hypotheses.push(Transform::Gravity {
            direction: Direction::Right,
        });

        // OutlineObjects: infer if training pairs have a consistent outline color
        if let Some(oc) = infer_outline_color(task) {
            hypotheses.push(Transform::OutlineObjects { outline_color: oc });
        }

        // CropToBoundingBox: try when output is smaller than input
        if let Some(pair) = task.train.first() {
            let ih = pair.input.len();
            let iw = if ih > 0 { pair.input[0].len() } else { 0 };
            let oh = pair.output.len();
            let ow = if oh > 0 { pair.output[0].len() } else { 0 };
            if oh < ih || ow < iw {
                hypotheses.push(Transform::CropToBoundingBox);
            }
        }

        // SwapColors: try all pairs of colors seen in first training pair
        if let Some(pair) = task.train.first() {
            let mut colors: Vec<u8> = pair.input.iter().flatten().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
            colors.sort();
            for i in 0..colors.len() {
                for j in (i+1)..colors.len() {
                    hypotheses.push(Transform::SwapColors { a: colors[i], b: colors[j] });
                }
            }
        }

        // RemoveSmallObjects: try min_size 1 (single-cell noise removal)
        hypotheses.push(Transform::RemoveSmallObjects { min_size: 2 });

        // ExtractLargestObject: try when output is smaller than input
        if let Some(pair) = task.train.first() {
            let ih = pair.input.len();
            let iw = if ih > 0 { pair.input[0].len() } else { 0 };
            let oh = pair.output.len();
            let ow = if oh > 0 { pair.output[0].len() } else { 0 };
            if oh <= ih && ow <= iw {
                hypotheses.push(Transform::ExtractLargestObject);
            }
        }

        // HollowObjects: try when output has interior cells removed
        {
            let feats = task.train.first().map(|p| analyze_grid(&p.output));
            if let Some(f) = feats {
                if f.colors.len() <= 3 {
                    // Try each non-bg color as hollow_color
                    for &c in &f.colors {
                        hypotheses.push(Transform::HollowObjects { hollow_color: c });
                    }
                }
            }
        }

        // RecolorBySize: when output has same objects but different colors
        if let Some(pair) = task.train.first() {
            if pair.input.len() == pair.output.len() {
                let out_feats = analyze_grid(&pair.output);
                let mut colors: Vec<u8> = out_feats.colors.iter()
                    .filter(|&&c| c != out_feats.background_color)
                    .copied().collect();
                colors.sort();
                if colors.len() >= 2 && colors.len() <= 5 {
                    hypotheses.push(Transform::RecolorBySize { color_order: colors });
                }
            }
        }

        // ConnectDots: when output has lines connecting same-color cells
        if let Some(pair) = task.train.first() {
            if pair.input.len() == pair.output.len() {
                let in_feats = analyze_grid(&pair.input);
                for &c in &in_feats.colors {
                    if c != in_feats.background_color {
                        hypotheses.push(Transform::ConnectDots { line_color: c });
                    }
                }
            }
        }

        // HalfGrid: try all four halves when output is half the size
        if let Some(pair) = task.train.first() {
            let ih = pair.input.len();
            let iw = if ih > 0 { pair.input[0].len() } else { 0 };
            let oh = pair.output.len();
            let ow = if oh > 0 { pair.output[0].len() } else { 0 };
            if (oh == ih / 2 && ow == iw) || (oh == ih && ow == iw / 2) || (oh <= ih && ow <= iw) {
                hypotheses.push(Transform::HalfGrid { half: GridHalf::Top });
                hypotheses.push(Transform::HalfGrid { half: GridHalf::Bottom });
                hypotheses.push(Transform::HalfGrid { half: GridHalf::Left });
                hypotheses.push(Transform::HalfGrid { half: GridHalf::Right });
            }
        }

        // Deduplicate
        let mut seen = HashSet::new();
        hypotheses.retain(|t| {
            let key = format!("{:?}", t);
            seen.insert(key)
        });

        hypotheses
    }

    fn infer_outline_color(task: &ArcTask) -> Option<u8> {
        // Find a consistent outline color across all training pairs.
        let mut outline_color: Option<u8> = None;
        for pair in &task.train {
            if pair.input.len() != pair.output.len() { return None; }
            let h = pair.input.len();
            if h == 0 { return None; }
            let w = pair.input[0].len();
            if pair.output[0].len() != w { return None; }
            let feats = analyze_grid(&pair.input);
            let bg = feats.background_color;
            let mut found_outline = false;
            for r in 0..h {
                for c in 0..w {
                    let ic = pair.input[r][c];
                    let oc = pair.output[r][c];
                    if ic == bg && oc != bg {
                        let adj = [(0i32,1i32),(0,-1),(1,0),(-1,0)];
                        let near = adj.iter().any(|(dr,dc)| {
                            let nr = r as i32 + dr; let nc = c as i32 + dc;
                            nr>=0 && nr<h as i32 && nc>=0 && nc<w as i32
                                && pair.input[nr as usize][nc as usize] != bg
                        });
                        if !near { return None; }
                        match outline_color {
                            None => { outline_color = Some(oc); found_outline = true; }
                            Some(prev) if prev != oc => return None,
                            _ => { found_outline = true; }
                        }
                    } else if ic != bg && oc != ic {
                        return None;
                    }
                }
            }
            if !found_outline { return None; }
        }
        outline_color
    }

    fn infer_color_map(task: &ArcTask) -> Option<HashMap<u8, u8>> {
        // Try to find a consistent color mapping across all training pairs
        let mut mapping: HashMap<u8, u8> = HashMap::new();
        for pair in &task.train {
            let ih = pair.input.len();
            let iw = if ih > 0 {
                pair.input[0].len()
            } else {
                continue;
            };
            let oh = pair.output.len();
            let ow = if oh > 0 {
                pair.output[0].len()
            } else {
                continue;
            };
            if ih != oh || iw != ow {
                return None;
            } // size must match
            for r in 0..ih {
                for c in 0..iw {
                    let ic = pair.input[r][c];
                    let oc = pair.output[r][c];
                    if let Some(&existing) = mapping.get(&ic) {
                        if existing != oc {
                            return None;
                        } // inconsistent
                    } else {
                        mapping.insert(ic, oc);
                    }
                }
            }
        }
        // Check it's not identity
        if mapping.iter().all(|(k, v)| k == v) {
            return None;
        }
        Some(mapping)
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 5  TRANSFORM APPLICATION
    // ═══════════════════════════════════════════════════════════════════

    pub fn apply_transform(input: &Grid, transform: &Transform) -> Option<Grid> {
        let h = input.len();
        if h == 0 {
            return None;
        }
        let w = input[0].len();

        match transform {
            Transform::Identity => Some(input.clone()),

            Transform::ColorMap(map) => {
                let mut out = input.clone();
                for row in &mut out {
                    for c in row {
                        if let Some(&mapped) = map.get(c) {
                            *c = mapped;
                        }
                    }
                }
                Some(out)
            }

            Transform::Scale { factor_h, factor_w } => {
                let mut out = vec![vec![0u8; w * factor_w]; h * factor_h];
                for r in 0..h {
                    for c in 0..w {
                        for dr in 0..*factor_h {
                            for dc in 0..*factor_w {
                                out[r * factor_h + dr][c * factor_w + dc] = input[r][c];
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::Tile { repeat_h, repeat_w } => {
                let mut out = vec![vec![0u8; w * repeat_w]; h * repeat_h];
                for r in 0..h * repeat_h {
                    for c in 0..w * repeat_w {
                        out[r][c] = input[r % h][c % w];
                    }
                }
                Some(out)
            }

            Transform::PatternTile => {
                // Each non-bg cell in input maps to a copy of input at that position
                // Output is input_h * input_h by input_w * input_w
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let oh = h * h;
                let ow = w * w;
                let mut out = vec![vec![bg; ow]; oh];
                for r in 0..h {
                    for c in 0..w {
                        if input[r][c] != bg {
                            // Place a copy of input at block position (r, c)
                            for ir in 0..h {
                                for ic in 0..w {
                                    let or = r * h + ir;
                                    let oc = c * w + ic;
                                    if or < oh && oc < ow {
                                        out[or][oc] = input[ir][ic];
                                    }
                                }
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::FlipHorizontal => {
                let mut out = input.clone();
                out.reverse();
                Some(out)
            }

            Transform::FlipVertical => {
                let out: Vec<Vec<u8>> = input
                    .iter()
                    .map(|row| {
                        let mut r = row.clone();
                        r.reverse();
                        r
                    })
                    .collect();
                Some(out)
            }

            Transform::Rotate90 => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[c][h - 1 - r] = input[r][c];
                    }
                }
                Some(out)
            }

            Transform::Rotate180 => {
                let mut out = vec![vec![0u8; w]; h];
                for r in 0..h {
                    for c in 0..w {
                        out[h - 1 - r][w - 1 - c] = input[r][c];
                    }
                }
                Some(out)
            }

            Transform::Rotate270 => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[w - 1 - c][r] = input[r][c];
                    }
                }
                Some(out)
            }

            Transform::Transpose => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[c][r] = input[r][c];
                    }
                }
                Some(out)
            }

            Transform::FillColor { from, to } => {
                let mut out = input.clone();
                for row in &mut out {
                    for c in row {
                        if *c == *from {
                            *c = *to;
                        }
                    }
                }
                Some(out)
            }

            Transform::FillEnclosed {
                boundary_color,
                fill_color,
            } => {
                let mut out = input.clone();
                // Find cells that are enclosed by boundary_color
                // Use flood fill from edges — anything not reached is enclosed
                let mut reachable = vec![vec![false; w]; h];
                let mut stack: Vec<(usize, usize)> = Vec::new();

                // Start from all edge cells that are NOT the boundary
                for r in 0..h {
                    for c in 0..w {
                        if (r == 0 || r == h - 1 || c == 0 || c == w - 1)
                            && input[r][c] != *boundary_color
                        {
                            if !reachable[r][c] {
                                reachable[r][c] = true;
                                stack.push((r, c));
                            }
                        }
                    }
                }

                while let Some((cr, cc)) = stack.pop() {
                    for (dr, dc) in &[(0i32, 1), (0, -1), (1, 0), (-1, 0)] {
                        let nr = cr as i32 + dr;
                        let nc = cc as i32 + dc;
                        if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if !reachable[nr][nc] && input[nr][nc] != *boundary_color {
                                reachable[nr][nc] = true;
                                stack.push((nr, nc));
                            }
                        }
                    }
                }

                // Fill unreachable non-boundary cells
                for r in 0..h {
                    for c in 0..w {
                        if !reachable[r][c] && out[r][c] != *boundary_color {
                            out[r][c] = *fill_color;
                        }
                    }
                }
                Some(out)
            }

            Transform::Gravity { direction } => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut out = vec![vec![bg; w]; h];

                match direction {
                    Direction::Down => {
                        for c in 0..w {
                            let mut dest = h;
                            for r in (0..h).rev() {
                                if input[r][c] != bg {
                                    dest -= 1;
                                    out[dest][c] = input[r][c];
                                }
                            }
                        }
                    }
                    Direction::Up => {
                        for c in 0..w {
                            let mut dest = 0;
                            for r in 0..h {
                                if input[r][c] != bg {
                                    out[dest][c] = input[r][c];
                                    dest += 1;
                                }
                            }
                        }
                    }
                    Direction::Right => {
                        for r in 0..h {
                            let mut dest = w;
                            for c in (0..w).rev() {
                                if input[r][c] != bg {
                                    dest -= 1;
                                    out[r][dest] = input[r][c];
                                }
                            }
                        }
                    }
                    Direction::Left => {
                        for r in 0..h {
                            let mut dest = 0;
                            for c in 0..w {
                                if input[r][c] != bg {
                                    out[r][dest] = input[r][c];
                                    dest += 1;
                                }
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::OutputSizeChange {
                height: oh,
                width: ow,
            } => {
                // Crop or pad to target size
                let mut out = vec![vec![0u8; *ow]; *oh];
                for r in 0..(*oh).min(h) {
                    for c in 0..(*ow).min(w) {
                        out[r][c] = input[r][c];
                    }
                }
                Some(out)
            }

            Transform::CopyInputToOutput => Some(input.clone()),

            Transform::OutlineObjects { outline_color } => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut out = input.clone();
                for r in 0..h {
                    for c in 0..w {
                        if input[r][c] == bg {
                            // Check if any 4-neighbor is a non-bg object cell
                            let adjacent = [(0i32,1),(0,-1),(1i32,0),(-1,0)];
                            let near_obj = adjacent.iter().any(|(dr,dc)| {
                                let nr = r as i32 + dr;
                                let nc = c as i32 + dc;
                                nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32
                                    && input[nr as usize][nc as usize] != bg
                            });
                            if near_obj {
                                out[r][c] = *outline_color;
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::CropToBoundingBox => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut min_r = h;
                let mut max_r = 0;
                let mut min_c = w;
                let mut max_c = 0;
                for r in 0..h {
                    for c in 0..w {
                        if input[r][c] != bg {
                            min_r = min_r.min(r);
                            max_r = max_r.max(r);
                            min_c = min_c.min(c);
                            max_c = max_c.max(c);
                        }
                    }
                }
                if min_r > max_r {
                    return None;
                }
                let out = (min_r..=max_r)
                    .map(|r| input[r][min_c..=max_c].to_vec())
                    .collect();
                Some(out)
            }

            Transform::SwapColors { a, b } => {
                let mut out = input.clone();
                for row in &mut out {
                    for c in row {
                        if *c == *a {
                            *c = *b;
                        } else if *c == *b {
                            *c = *a;
                        }
                    }
                }
                Some(out)
            }

            Transform::RemoveSmallObjects { min_size } => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut visited = vec![vec![false; w]; h];
                let mut out = input.clone();
                for sr in 0..h {
                    for sc in 0..w {
                        if !visited[sr][sc] && input[sr][sc] != bg {
                            // flood fill to get component
                            let mut cells = Vec::new();
                            let mut stack = vec![(sr, sc)];
                            visited[sr][sc] = true;
                            while let Some((cr, cc)) = stack.pop() {
                                cells.push((cr, cc));
                                for (dr, dc) in &[(0i32,1),(0,-1),(1i32,0),(-1,0)] {
                                    let nr = cr as i32 + dr;
                                    let nc = cc as i32 + dc;
                                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                                        let (nr, nc) = (nr as usize, nc as usize);
                                        if !visited[nr][nc] && input[nr][nc] != bg {
                                            visited[nr][nc] = true;
                                            stack.push((nr, nc));
                                        }
                                    }
                                }
                            }
                            if cells.len() < *min_size {
                                for (r, c) in cells {
                                    out[r][c] = bg;
                                }
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::MostCommonObjectColor => {
                // Return 1x1 grid with the most common non-background color
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut color_counts: HashMap<u8, usize> = HashMap::new();
                for row in input {
                    for &c in row {
                        if c != bg {
                            *color_counts.entry(c).or_insert(0) += 1;
                        }
                    }
                }
                let most_common = color_counts.into_iter().max_by_key(|(_, cnt)| *cnt)?.0;
                Some(vec![vec![most_common]])
            }

            Transform::ConditionalColorSwap {
                condition_color,
                swap,
            } => {
                // For each cell of condition_color, swap adjacent cells per the swap map
                let mut out = input.clone();
                for r in 0..h {
                    for c in 0..w {
                        if input[r][c] == *condition_color {
                            if let Some(&new_color) = swap.get(&input[r][c]) {
                                out[r][c] = new_color;
                            }
                        } else if let Some(&new_color) = swap.get(&input[r][c]) {
                            out[r][c] = new_color;
                        }
                    }
                }
                Some(out)
            }

            Transform::ExtractLargestObject => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut visited = vec![vec![false; w]; h];
                let mut best_cells: Vec<(usize, usize)> = Vec::new();
                for sr in 0..h {
                    for sc in 0..w {
                        if !visited[sr][sc] && input[sr][sc] != bg {
                            let mut cells = Vec::new();
                            let mut stack = vec![(sr, sc)];
                            visited[sr][sc] = true;
                            while let Some((cr, cc)) = stack.pop() {
                                cells.push((cr, cc));
                                for (dr, dc) in &[(0i32,1),(0,-1),(1i32,0),(-1,0)] {
                                    let nr = cr as i32 + dr;
                                    let nc = cc as i32 + dc;
                                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                                        let (nr, nc) = (nr as usize, nc as usize);
                                        if !visited[nr][nc] && input[nr][nc] != bg {
                                            visited[nr][nc] = true;
                                            stack.push((nr, nc));
                                        }
                                    }
                                }
                            }
                            if cells.len() > best_cells.len() {
                                best_cells = cells;
                            }
                        }
                    }
                }
                if best_cells.is_empty() { return None; }
                let min_r = best_cells.iter().map(|(r,_)| *r).min().unwrap();
                let max_r = best_cells.iter().map(|(r,_)| *r).max().unwrap();
                let min_c = best_cells.iter().map(|(_,c)| *c).min().unwrap();
                let max_c = best_cells.iter().map(|(_,c)| *c).max().unwrap();
                let out = (min_r..=max_r)
                    .map(|r| input[r][min_c..=max_c].to_vec())
                    .collect();
                Some(out)
            }

            Transform::HollowObjects { hollow_color } => {
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut out = input.clone();
                for r in 0..h {
                    for c in 0..w {
                        if input[r][c] == *hollow_color {
                            // Cell is interior if all 4 neighbors are non-bg
                            let is_interior = [(0i32,1),(0,-1),(1i32,0),(-1,0)].iter().all(|(dr,dc)| {
                                let nr = r as i32 + dr;
                                let nc = c as i32 + dc;
                                nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32
                                    && input[nr as usize][nc as usize] != bg
                            });
                            if is_interior {
                                out[r][c] = bg;
                            }
                        }
                    }
                }
                Some(out)
            }

            Transform::RecolorBySize { color_order } => {
                // Find all connected objects, sort by size, assign colors in order
                let feats = analyze_grid(input);
                let bg = feats.background_color;
                let mut visited = vec![vec![false; w]; h];
                let mut objects: Vec<Vec<(usize, usize)>> = Vec::new();
                for sr in 0..h {
                    for sc in 0..w {
                        if !visited[sr][sc] && input[sr][sc] != bg {
                            let mut cells = Vec::new();
                            let mut stack = vec![(sr, sc)];
                            visited[sr][sc] = true;
                            while let Some((cr, cc)) = stack.pop() {
                                cells.push((cr, cc));
                                for (dr, dc) in &[(0i32,1),(0,-1),(1i32,0),(-1,0)] {
                                    let nr = cr as i32 + dr;
                                    let nc = cc as i32 + dc;
                                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                                        let (nr, nc) = (nr as usize, nc as usize);
                                        if !visited[nr][nc] && input[nr][nc] != bg {
                                            visited[nr][nc] = true;
                                            stack.push((nr, nc));
                                        }
                                    }
                                }
                            }
                            objects.push(cells);
                        }
                    }
                }
                // Sort smallest→largest
                objects.sort_by_key(|o| o.len());
                let mut out = vec![vec![bg; w]; h];
                for (i, obj) in objects.iter().enumerate() {
                    let color = color_order.get(i).copied().unwrap_or(*color_order.last().unwrap_or(&1));
                    for &(r, c) in obj {
                        out[r][c] = color;
                    }
                }
                Some(out)
            }

            Transform::ConnectDots { line_color } => {
                // Draw straight horizontal+vertical lines connecting all cells of line_color
                let mut out = input.clone();
                // Collect all positions of line_color
                let dots: Vec<(usize, usize)> = (0..h)
                    .flat_map(|r| (0..w).filter_map(move |c| {
                        if input[r][c] == *line_color { Some((r, c)) } else { None }
                    }))
                    .collect();
                // Connect dots sharing the same row or column
                for i in 0..dots.len() {
                    for j in (i+1)..dots.len() {
                        let (r1, c1) = dots[i];
                        let (r2, c2) = dots[j];
                        if r1 == r2 {
                            let (cmin, cmax) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                            for c in cmin..=cmax { out[r1][c] = *line_color; }
                        } else if c1 == c2 {
                            let (rmin, rmax) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
                            for r in rmin..=rmax { out[r][c1] = *line_color; }
                        }
                    }
                }
                Some(out)
            }

            Transform::HalfGrid { half } => {
                match half {
                    GridHalf::Top => {
                        let half_h = (h + 1) / 2;
                        Some(input[..half_h].to_vec())
                    }
                    GridHalf::Bottom => {
                        let half_h = h / 2;
                        Some(input[half_h..].to_vec())
                    }
                    GridHalf::Left => {
                        let half_w = (w + 1) / 2;
                        Some(input.iter().map(|row| row[..half_w].to_vec()).collect())
                    }
                    GridHalf::Right => {
                        let half_w = w / 2;
                        Some(input.iter().map(|row| row[half_w..].to_vec()).collect())
                    }
                }
            }

            _ => None,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 6  GRID ENCODER — Maps grid features to layer confidence signals
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, Clone)]
    pub struct LayerConfidences {
        pub base_physics: f32,       // L1: structural regularity
        pub extended_physics: f32,   // L2: transformation complexity
        pub cross_domain: f32,       // L3: cross-pair consistency
        pub gaia_consciousness: f32, // L4: pattern intuition
        pub multilingual: f32,       // L5: representational diversity
        pub collaborative: f32,      // L6: multi-example agreement
        pub external_apis: f32,      // L7: validation confidence
        pub pre_cognitive: f32,      // L8: outcome prediction confidence
    }

    /// Encode layer confidences for a specific candidate (braid-aware, per-candidate).
    pub fn encode_candidate_confidences(
        task: &ArcTask,
        candidate: &TransformCandidate,
        all_candidates: &[TransformCandidate],
    ) -> LayerConfidences {
        // Use this candidate's accuracy as the primary signal
        let cand_conf = candidate.confidence;
        let best_conf = all_candidates.first().map(|c| c.confidence).unwrap_or(0.0);
        let num_perfect = all_candidates.iter().filter(|c| c.confidence >= 1.0).count();
        let num_candidates = all_candidates.len();

        let train_features: Vec<_> = task
            .train
            .iter()
            .map(|p| (analyze_grid(&p.input), analyze_grid(&p.output)))
            .collect();

        let avg_symmetry: f32 = train_features
            .iter()
            .map(|(inf, _)| {
                let s = &inf.symmetry;
                let count = [s.horizontal, s.vertical, s.diagonal, s.rotational_90]
                    .iter()
                    .filter(|&&x| x)
                    .count();
                count as f32 / 4.0
            })
            .sum::<f32>()
            / train_features.len().max(1) as f32;

        let size_consistent = train_features.windows(2).all(|w| {
            let r0 = (
                w[0].1.height as f32 / w[0].0.height.max(1) as f32,
                w[0].1.width as f32 / w[0].0.width.max(1) as f32,
            );
            let r1 = (
                w[1].1.height as f32 / w[1].0.height.max(1) as f32,
                w[1].1.width as f32 / w[1].0.width.max(1) as f32,
            );
            (r0.0 - r1.0).abs() < 0.01 && (r0.1 - r1.1).abs() < 0.01
        });

        // L1: structural regularity + this candidate's training accuracy
        let base_physics = (0.2 + avg_symmetry * 0.3 + cand_conf * 0.5).min(1.0);

        // L2: transformation complexity — perfect candidate = simple transform
        let extended_physics = if cand_conf >= 1.0 {
            0.95
        } else if cand_conf > 0.5 {
            0.6 + cand_conf * 0.3
        } else {
            cand_conf * 0.6
        };

        // L3: cross-domain consistency weighted by candidate quality
        let cross_domain = if size_consistent { 0.75 } else { 0.35 } + cand_conf * 0.25;

        // L4: pattern intuition — how unique/confident is this candidate vs others
        let rank_factor = if best_conf > 0.0 { cand_conf / best_conf } else { 0.0 };
        let color_consistency = train_features
            .windows(2)
            .all(|w| w[0].0.colors == w[1].0.colors);
        let gaia_consciousness =
            (if color_consistency { 0.5 } else { 0.2 } + rank_factor * 0.4 + cand_conf * 0.1)
                .min(1.0);

        // L5: diversity of representations
        let avg_colors: f32 = train_features
            .iter()
            .map(|(inf, _)| inf.colors.len() as f32)
            .sum::<f32>()
            / train_features.len().max(1) as f32;
        let multilingual = (avg_colors / 10.0).min(1.0) * 0.4 + cand_conf * 0.4 + 0.1;

        // L6: agreement — does this candidate agree across all training pairs?
        let collaborative = if cand_conf >= 1.0 {
            0.9
        } else {
            0.3 + cand_conf * 0.6
        };

        // L7: external validation — how many candidates at full confidence?
        let external_apis = if cand_conf >= 1.0 {
            0.85 + (num_perfect as f32 / 20.0).min(0.15)
        } else if num_candidates > 0 {
            0.25 + cand_conf * 0.4
        } else {
            0.1
        };

        // L8: pre-cognitive outcome prediction — direct candidate confidence
        let pre_cognitive = cand_conf * 0.8 + if cand_conf >= 1.0 { 0.2 } else { 0.0 };

        LayerConfidences {
            base_physics: base_physics.min(1.0),
            extended_physics: extended_physics.min(1.0),
            cross_domain: cross_domain.min(1.0),
            gaia_consciousness: gaia_consciousness.min(1.0),
            multilingual: multilingual.min(1.0),
            collaborative: collaborative.min(1.0),
            external_apis: external_apis.min(1.0),
            pre_cognitive: pre_cognitive.min(1.0),
        }
    }

    pub fn encode_task_to_confidences(
        task: &ArcTask,
        candidates: &[TransformCandidate],
    ) -> LayerConfidences {
        let best_conf = candidates.first().map(|c| c.confidence).unwrap_or(0.0);
        let num_perfect = candidates.iter().filter(|c| c.confidence >= 1.0).count();
        let num_candidates = candidates.len();

        // Analyze structural features
        let train_features: Vec<_> = task
            .train
            .iter()
            .map(|p| (analyze_grid(&p.input), analyze_grid(&p.output)))
            .collect();

        // L1: Base Physics — structural regularity of input grids
        let avg_symmetry: f32 = train_features
            .iter()
            .map(|(inf, _)| {
                let s = &inf.symmetry;
                let count = [s.horizontal, s.vertical, s.diagonal, s.rotational_90]
                    .iter()
                    .filter(|&&x| x)
                    .count();
                count as f32 / 4.0
            })
            .sum::<f32>()
            / train_features.len().max(1) as f32;
        let base_physics = 0.3 + avg_symmetry * 0.4 + if best_conf > 0.0 { 0.3 } else { 0.0 };

        // L2: Extended Physics — transformation complexity (simpler = higher)
        let extended_physics = if num_perfect > 0 {
            0.9 - (num_perfect as f32 - 1.0).min(5.0) * 0.05
        } else if best_conf > 0.5 {
            0.6
        } else {
            0.3
        };

        // L3: Cross-Domain — consistency of input→output size relationships
        let size_consistent = train_features.windows(2).all(|w| {
            let r0 = (
                w[0].1.height as f32 / w[0].0.height.max(1) as f32,
                w[0].1.width as f32 / w[0].0.width.max(1) as f32,
            );
            let r1 = (
                w[1].1.height as f32 / w[1].0.height.max(1) as f32,
                w[1].1.width as f32 / w[1].0.width.max(1) as f32,
            );
            (r0.0 - r1.0).abs() < 0.01 && (r0.1 - r1.1).abs() < 0.01
        });
        let cross_domain = if size_consistent { 0.8 } else { 0.4 } + best_conf * 0.2;

        // L4: GAIA Consciousness — intuitive pattern recognition
        let color_consistency = train_features
            .windows(2)
            .all(|w| w[0].0.colors == w[1].0.colors);
        let gaia_consciousness = if color_consistency { 0.6 } else { 0.3 } + best_conf * 0.3;

        // L5: Multilingual — diversity of representations / colors used
        let avg_colors: f32 = train_features
            .iter()
            .map(|(inf, _)| inf.colors.len() as f32)
            .sum::<f32>()
            / train_features.len().max(1) as f32;
        let multilingual = (avg_colors / 10.0).min(1.0) * 0.5 + best_conf * 0.3 + 0.1;

        // L6: Collaborative — agreement across multiple training examples
        let collaborative = if num_perfect > 0 {
            0.5 + (best_conf * 0.5)
        } else {
            0.2 + (best_conf * 0.3)
        };

        // L7: External — validation confidence (how many candidates match all?)
        let external_apis = if num_perfect > 0 {
            0.8 + (num_perfect as f32 / 20.0).min(0.2)
        } else if num_candidates > 0 {
            0.3 + best_conf * 0.3
        } else {
            0.1
        };

        // L8: Pre-cognitive — outcome prediction confidence
        let pre_cognitive = best_conf * 0.7 + if num_perfect > 0 { 0.3 } else { 0.0 };

        LayerConfidences {
            base_physics: base_physics.min(1.0),
            extended_physics: extended_physics.min(1.0),
            cross_domain: cross_domain.min(1.0),
            gaia_consciousness: gaia_consciousness.min(1.0),
            multilingual: multilingual.min(1.0),
            collaborative: collaborative.min(1.0),
            external_apis: external_apis.min(1.0),
            pre_cognitive: pre_cognitive.min(1.0),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 7  SOLVER — Uses layer system to rank and select candidates
    // ═══════════════════════════════════════════════════════════════════

    pub struct ArcSolver {
        stack: LayerStack,
        ttt_bridge: Option<TTTBridge>,
    }

    impl ArcSolver {
        pub fn new() -> Self {
            let config = LayerStackConfig::new()
                .with_global_amplification(1.15)
                .with_max_iterations(8)
                .with_max_confidence(2.0)
                .with_amplification_damping(0.8)
                .with_all_phases(); // Phase 3 + 5 + 6

            let mut stack = LayerStack::with_config(config);

            // Register all 15 bridges
            let network = BridgeBuilder::new()
                .with_all_extended_bridges()
                .with_global_amplification(1.15)
                .build();
            for bridge in network.bridges() {
                stack.register_bridge(bridge.clone());
            }

            ArcSolver {
                stack,
                ttt_bridge: None,
            }
        }

        pub fn with_ttt(mut self, script_path: PathBuf) -> Self {
            self.ttt_bridge = Some(TTTBridge::new(script_path));
            self
        }

        pub fn solve(
            &mut self,
            task: &ArcTask,
            max_trials: usize,
            task_json: Option<&serde_json::Value>,
            verbose: bool,
        ) -> Vec<(Grid, f32)> {
            // 1. Detect candidate transforms
            let candidates = detect_transforms(task);

            // PRE-COG TRAINING: feed the visualization engine known-correct signals from
            // training pairs BEFORE we score test candidates. For each training pair, find
            // which candidate produces the exact correct output and run it through the stack
            // so that record_outcome receives a high-confidence training signal.
            for train_pair in &task.train {
                for candidate in candidates.iter().take(max_trials.max(3)) {
                    if let Some(output) = apply_transform(&train_pair.input, &candidate.transform) {
                        let is_correct = output == train_pair.output;
                        // Only train on correct candidates (positive examples only)
                        // so the visualization engine calibrates toward what works.
                        if is_correct {
                            let cand_conf = encode_candidate_confidences(task, candidate, &candidates);
                            // Boost pre_cognitive toward 1.0 on correct candidates
                            let mut boosted = cand_conf.clone();
                            boosted.pre_cognitive = 0.95;
                            let train_state = LayerState::with_confidence(
                                Layer::BasePhysics,
                                boosted.clone(),
                                boosted.base_physics,
                            );
                            self.stack.set_difficulty_hint(Some(0.05)); // very easy — correct answer
                            self.stack.clear_states();
                            let _ = self.stack.process_bidirectional(train_state);
                            break; // one correct candidate per training pair is enough
                        }
                    }
                }
            }

            // 2. Process through layer system — OCTO Braid scores each candidate individually
            let mut all_results = Vec::new();

            for (test_idx, test_pair) in task.test.iter().enumerate() {
                let mut trial_results: Vec<(Grid, f32)> = Vec::new();

                // Generate candidate outputs — OCTO Braid scores each candidate individually
                for candidate in candidates.iter().take(max_trials.max(3)) {
                    if let Some(output) = apply_transform(&test_pair.input, &candidate.transform) {

                        // Per-candidate difficulty: perfect match = easy for braid,
                        // partial match = hard — braid restricts cap accordingly
                        let cand_difficulty = if candidate.confidence >= 1.0 {
                            0.1
                        } else if candidate.confidence > 0.75 {
                            0.3
                        } else if candidate.confidence > 0.5 {
                            0.5
                        } else {
                            0.75 + (0.5 - candidate.confidence).max(0.0) * 0.4
                        };
                        self.stack.set_difficulty_hint(Some(cand_difficulty));

                        // Seed all 8 layers with per-candidate confidences
                        let cand_confidences = encode_candidate_confidences(
                            task, candidate, &candidates,
                        );

                        let input_state = LayerState::with_confidence(
                            Layer::BasePhysics,
                            cand_confidences.clone(),
                            cand_confidences.base_physics,
                        );

                        self.stack.clear_states();
                        let result = self.stack.process_bidirectional(input_state);

                        // Extract live braid signals for this candidate
                        let braid = self.stack.octo_braid()
                            .map(|b| b.last_signals().clone())
                            .unwrap_or_default();

                        // OCTO Braid-driven score:
                        // effective_cap  → how much the braid trusts this candidate (normalized [0,1])
                        // temperature    → uncertainty penalty (lower = better)
                        // system2_active → extra scrutiny flag (slight penalty)
                        // combined_confidence → layer stack's multiplicative amplification signal
                        // visualization_confidence → pre-cognitive prediction calibrated by fidelity
                        let braid_cap_score = braid.effective_cap / 2.0;
                        let temp_penalty = 1.0 / (1.0 + braid.temperature * 0.3);
                        let sys2_penalty = if braid.system2_active { 0.9 } else { 1.0 };
                        let layer_score = (result.combined_confidence / 2.0).min(1.0);
                        let viz_score = result.visualization
                            .as_ref()
                            .map(|v| v.visualization_confidence)
                            .unwrap_or(0.5);

                        if verbose {
                            eprintln!(
                                "      braid: cap={:.3} temp={:.3} sys2={} diff={:.2} | cand={:.3} viz={:.3}",
                                braid.effective_cap,
                                braid.temperature,
                                braid.system2_active,
                                cand_difficulty,
                                candidate.confidence,
                                viz_score,
                            );
                        }

                        // Primary:   candidate training accuracy (ground truth)       50%
                        // Secondary: braid cap × temp × sys2 (live OCTO signal)       25%
                        // Tertiary:  pre-cog visualization (fidelity-calibrated)       15%
                        // Layer:     stack multiplicative output                       10%
                        let combined_score = candidate.confidence * 0.50
                            + braid_cap_score * temp_penalty * sys2_penalty * 0.25
                            + viz_score * 0.15
                            + layer_score * 0.10;

                        trial_results.push((output, combined_score));
                    }
                }

                // 5. TTT FALLBACK — If brute-force didn't find a perfect match,
                //    use the LLM TTT solver with OCTO Braid signal integration.
                let brute_force_perfect = trial_results
                    .first()
                    .map(|(_, score)| *score >= 0.99)
                    .unwrap_or(false);

                if !brute_force_perfect {
                    if let Some(ref ttt_bridge) = self.ttt_bridge {
                        if let Some(task_json) = task_json {
                            if verbose {
                                eprintln!(
                                    "    TTT: Brute-force best={:.3}, invoking LLM solver...",
                                    trial_results.first().map(|(_, s)| *s).unwrap_or(0.0)
                                );
                            }

                            // Extract current braid signals from stack
                            let braid_signals = TTTBridge::extract_braid_signals(&self.stack);

                            // Call TTT solver subprocess
                            if let Some((ttt_grids, ttt_conf, feedback)) =
                                ttt_bridge.solve_test(task_json, test_idx, braid_signals, verbose)
                            {
                                // Feed braid feedback back into the layer stack
                                TTTBridge::apply_feedback_to_stack(&mut self.stack, &feedback);

                                // Re-process through layer system with updated braid state
                                let ttt_input_state = LayerState::with_confidence(
                                    Layer::BasePhysics,
                                    ttt_conf as f32,
                                    // TTT confidence drives the input seed
                                    ttt_conf as f32 * 0.9,
                                );
                                self.stack.clear_states();
                                let ttt_result = self.stack.process_bidirectional(ttt_input_state);
                                let ttt_layer_conf = ttt_result.combined_confidence / 2.0;

                                // Score TTT predictions using combined TTT + layer confidence
                                for (i, grid) in ttt_grids.into_iter().enumerate() {
                                    // First TTT prediction gets full confidence, subsequent get less
                                    let rank_penalty = 1.0 / (1.0 + i as f32 * 0.3);
                                    let ttt_score = (ttt_conf as f32 * 0.5
                                        + ttt_layer_conf * 0.3
                                        + 0.2) // base bonus for attempting TTT
                                        * rank_penalty;

                                    trial_results.push((grid, ttt_score));
                                }

                                if verbose {
                                    eprintln!(
                                        "    TTT: Added {} predictions, braid feedback applied (difficulty={:.3})",
                                        ttt_conf,
                                        1.0 - feedback.ttt_confidence as f32
                                    );
                                }
                            }
                        }
                    }
                }

                // If no candidates worked, try identity as fallback
                if trial_results.is_empty() {
                    trial_results.push((test_pair.input.clone(), 0.01));
                }

                // Sort by score and take top result
                trial_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                trial_results.truncate(max_trials);

                all_results.extend(trial_results);
            }

            all_results
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 7.5  TTT BRIDGE — Subprocess integration with Python LLM solver
    // ═══════════════════════════════════════════════════════════════════

    /// Serializable braid signals to pass to the Python TTT solver.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TTTBraidSignals {
        temperature: f32,
        cap_aggressiveness: f32,
        resonance_sensitivity: f32,
        global_damping: f32,
        system2_active: bool,
    }

    /// Request sent to ttt_solver.py via stdin.
    #[derive(Debug, Clone, Serialize)]
    struct TTTRequest {
        task: serde_json::Value, // raw ARC task JSON
        test_index: usize,
        config: TTTConfig,
        braid_signals: TTTBraidSignals,
    }

    /// Config for the TTT solver.
    #[derive(Debug, Clone, Serialize)]
    struct TTTConfig {
        max_refinement_iters: usize,
        max_tokens: usize,
        temperature: f32,
        enable_airv: bool,
        enable_self_check: bool,
        max_augmentations: usize,
    }

    /// Response from ttt_solver.py via stdout.
    #[derive(Debug, Clone, Deserialize)]
    struct TTTResponse {
        predictions: Vec<Vec<Vec<u8>>>,
        confidence: f64,
        reasoning: String,
        augmentations_used: usize,
        refinement_iterations: usize,
        braid_signals: TTTBraidFeedback,
        elapsed_ms: f64,
    }

    /// Braid feedback signals returned from TTT solver.
    #[derive(Debug, Clone, Deserialize)]
    struct TTTBraidFeedback {
        ttt_confidence: f64,
        ttt_temperature: f64,
        ttt_augmentations: usize,
        ttt_refinements: usize,
        ttt_predictions_count: usize,
        ttt_agreement: f64,
        ttt_elapsed_ms: f64,
        suggested_reserve_scale: f64,
        suggested_cap: f64,
        suggested_system2: bool,
    }

    /// Bridge to the Python TTT solver subprocess.
    pub struct TTTBridge {
        python_path: String,
        script_path: PathBuf,
        enable_airv: bool,
        max_refinement_iters: usize,
    }

    impl TTTBridge {
        pub fn new(script_path: PathBuf) -> Self {
            TTTBridge {
                python_path: "/usr/local/bin/python3".to_string(),
                script_path,
                enable_airv: true,
                max_refinement_iters: 2,
            }
        }

        /// Extract current braid signals from the LayerStack.
        fn extract_braid_signals(stack: &LayerStack) -> TTTBraidSignals {
            if let Some(braid) = stack.octo_braid() {
                let signals = braid.last_signals();
                TTTBraidSignals {
                    temperature: signals.temperature,
                    cap_aggressiveness: signals.cap_aggressiveness,
                    resonance_sensitivity: signals.resonance_sensitivity,
                    global_damping: signals.global_damping,
                    system2_active: signals.system2_active,
                }
            } else {
                // Default signals when braid is not active
                TTTBraidSignals {
                    temperature: 0.3,
                    cap_aggressiveness: 1.0,
                    resonance_sensitivity: 1.0,
                    global_damping: 0.8,
                    system2_active: true,
                }
            }
        }

        /// Call the TTT solver for a specific test input.
        /// Returns (predictions, confidence, braid_feedback).
        fn solve_test(
            &self,
            task_json: &serde_json::Value,
            test_index: usize,
            braid_signals: TTTBraidSignals,
            verbose: bool,
        ) -> Option<(Vec<Grid>, f64, TTTBraidFeedback)> {
            let config = TTTConfig {
                max_refinement_iters: self.max_refinement_iters,
                max_tokens: 1024,
                temperature: braid_signals.temperature * 0.3, // scale down for LLM
                enable_airv: self.enable_airv && braid_signals.system2_active,
                enable_self_check: braid_signals.system2_active,
                max_augmentations: (4.0 * braid_signals.resonance_sensitivity) as usize,
            };

            let request = TTTRequest {
                task: task_json.clone(),
                test_index,
                config,
                braid_signals,
            };

            let request_json = match serde_json::to_string(&request) {
                Ok(j) => j,
                Err(e) => {
                    if verbose {
                        eprintln!("    TTT: Failed to serialize request: {}", e);
                    }
                    return None;
                }
            };

            // Spawn Python subprocess
            let start = Instant::now();
            let mut child = match Command::new(&self.python_path)
                .arg(&self.script_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    if verbose {
                        eprintln!("    TTT: Failed to spawn Python: {}", e);
                    }
                    return None;
                }
            };

            // Write request to stdin
            if let Some(ref mut stdin) = child.stdin {
                if stdin.write_all(request_json.as_bytes()).is_err() {
                    if verbose {
                        eprintln!("    TTT: Failed to write to stdin");
                    }
                    return None;
                }
            }
            // Drop stdin to signal EOF
            drop(child.stdin.take());

            // Wait for result (with timeout — model loading can be slow)
            let output = match child.wait_with_output() {
                Ok(o) => o,
                Err(e) => {
                    if verbose {
                        eprintln!("    TTT: Subprocess error: {}", e);
                    }
                    return None;
                }
            };

            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            if !output.status.success() {
                if verbose {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("    TTT: Process exited with error: {}", stderr);
                }
                return None;
            }

            // Parse response
            let stdout = String::from_utf8_lossy(&output.stdout);
            let response: TTTResponse = match serde_json::from_str(&stdout) {
                Ok(r) => r,
                Err(e) => {
                    if verbose {
                        eprintln!(
                            "    TTT: Failed to parse response: {} (raw: {})",
                            e,
                            &stdout[..stdout.len().min(200)]
                        );
                    }
                    return None;
                }
            };

            if verbose {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    eprintln!("    TTT stderr: {}", stderr.trim());
                }
                eprintln!(
                    "    TTT: {} predictions, confidence={:.3}, airv={}, refinements={}, {:.0}ms",
                    response.predictions.len(),
                    response.confidence,
                    response.augmentations_used,
                    response.refinement_iterations,
                    elapsed
                );
            }

            if response.predictions.is_empty() {
                return None;
            }

            // Convert predictions (Vec<Vec<u8>> → Grid)
            let grids: Vec<Grid> = response.predictions;

            Some((grids, response.confidence, response.braid_signals))
        }

        /// Feed TTT braid feedback back into the LayerStack.
        fn apply_feedback_to_stack(stack: &mut LayerStack, feedback: &TTTBraidFeedback) {
            // Use TTT confidence to adjust difficulty hint
            // Low TTT confidence → harder task → increase difficulty
            let new_difficulty = 1.0 - feedback.ttt_confidence as f32;
            stack.set_difficulty_hint(Some(new_difficulty.clamp(0.1, 0.95)));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 8  EVALUATION HARNESS
    // ═══════════════════════════════════════════════════════════════════

    pub fn evaluate_task(
        solver: &mut ArcSolver,
        task_id: &str,
        task: &ArcTask,
        task_json: Option<&serde_json::Value>,
        max_trials: usize,
        verbose: bool,
    ) -> TaskResult {
        let start = Instant::now();

        let predictions = solver.solve(task, max_trials, task_json, verbose);

        let mut correct = 0;
        let mut total = 0;
        let mut trials_used = Vec::new();

        let mut pred_idx = 0;
        for (test_idx, test_pair) in task.test.iter().enumerate() {
            total += 1;
            let mut solved = false;
            let mut trial_count = 0;

            for trial in 0..max_trials {
                if pred_idx + trial >= predictions.len() {
                    break;
                }
                trial_count += 1;
                let (ref predicted, score) = predictions[pred_idx + trial];
                if *predicted == test_pair.output {
                    solved = true;
                    if verbose {
                        println!(
                            "    Test {}: CORRECT on trial {} (score={:.3})",
                            test_idx + 1,
                            trial + 1,
                            score
                        );
                    }
                    break;
                } else if verbose {
                    let ph = predicted.len();
                    let pw = if ph > 0 { predicted[0].len() } else { 0 };
                    let eh = test_pair.output.len();
                    let ew = if eh > 0 { test_pair.output[0].len() } else { 0 };
                    let size_match = ph == eh && pw == ew;
                    let cell_match = if size_match {
                        let total_cells = eh * ew;
                        let matching = (0..eh)
                            .map(|r| {
                                (0..ew)
                                    .filter(|&c| predicted[r][c] == test_pair.output[r][c])
                                    .count()
                            })
                            .sum::<usize>();
                        format!("{}/{} cells", matching, total_cells)
                    } else {
                        format!("size {}x{} vs {}x{}", ph, pw, eh, ew)
                    };
                    println!(
                        "    Test {}: trial {} WRONG (score={:.3}, {})",
                        test_idx + 1,
                        trial + 1,
                        score,
                        cell_match
                    );
                }
            }
            if solved {
                correct += 1;
            }
            trials_used.push(trial_count);
            pred_idx += max_trials.min(predictions.len() - pred_idx);
        }

        let confidence = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        TaskResult {
            task_id: task_id.to_string(),
            correct,
            total,
            trials_used,
            confidence,
            time_ms,
        }
    }

    pub fn evaluate_all(
        data_dir: &Path,
        split: &str,
        limit: Option<usize>,
        single_task: Option<&str>,
        max_trials: usize,
        verbose: bool,
        enable_ttt: bool,
    ) -> Result<EvalSummary, String> {
        let tasks = load_tasks(data_dir, split)?;
        let solver_base = ArcSolver::new();
        let mut solver = if enable_ttt {
            // Look for ttt_solver.py relative to the binary
            let script_path = PathBuf::from("ttt_solver.py");
            if !script_path.exists() {
                return Err(format!(
                    "TTT enabled but {} not found. Place ttt_solver.py in the working directory.",
                    script_path.display()
                ));
            }
            solver_base.with_ttt(script_path)
        } else {
            solver_base
        };

        let tasks_to_eval: Vec<_> = if let Some(task_id) = single_task {
            tasks.into_iter().filter(|(id, _)| id == task_id).collect()
        } else if let Some(limit) = limit {
            tasks.into_iter().take(limit).collect()
        } else {
            tasks
        };

        println!("================================================================");
        println!("     RustyWorm ARC-AGI-2 EVALUATION");
        println!("================================================================");
        println!(
            "  Split: {}  |  Tasks: {}  |  Trials: {}",
            split,
            tasks_to_eval.len(),
            max_trials
        );
        println!(
            "  Engine: 8-Layer Multiplicative + Phase 3/5/6 (OCTO Braid){}",
            if enable_ttt { " + TTT (LLM)" } else { "" }
        );
        println!("================================================================\n");

        let mut total_tasks = 0;
        let mut tasks_solved = 0;
        let mut total_test_inputs = 0;
        let mut test_inputs_solved = 0;
        let mut total_confidence = 0.0;
        let mut total_time = 0.0;
        let mut per_category: HashMap<String, (usize, usize)> = HashMap::new();

        let eval_start = Instant::now();

        for (idx, (task_id, task)) in tasks_to_eval.iter().enumerate() {
            if verbose {
                println!("[{}/{}] Task {}:", idx + 1, tasks_to_eval.len(), task_id);
                println!(
                    "  Train pairs: {}  Test pairs: {}",
                    task.train.len(),
                    task.test.len()
                );
                if let Some(p) = task.train.first() {
                    let ih = p.input.len();
                    let iw = if ih > 0 { p.input[0].len() } else { 0 };
                    let oh = p.output.len();
                    let ow = if oh > 0 { p.output[0].len() } else { 0 };
                    println!("  Input: {}x{}  Output: {}x{}", ih, iw, oh, ow);
                }
            }

            // Load raw task JSON for TTT bridge (if enabled)
            let task_json: Option<serde_json::Value> = if enable_ttt {
                let task_path = data_dir.join(split).join(format!("{}.json", task_id));
                fs::read_to_string(&task_path)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
            } else {
                None
            };

            let result = evaluate_task(
                &mut solver,
                task_id,
                task,
                task_json.as_ref(),
                max_trials,
                verbose,
            );

            total_tasks += 1;
            total_test_inputs += result.total;
            test_inputs_solved += result.correct;
            total_confidence += result.confidence;
            total_time += result.time_ms;

            let task_solved = result.correct == result.total;
            if task_solved {
                tasks_solved += 1;
            }

            // Categorize by input/output size relationship
            let category = if let Some(p) = task.train.first() {
                let (ih, iw) = (p.input.len(), p.input.get(0).map_or(0, |r| r.len()));
                let (oh, ow) = (p.output.len(), p.output.get(0).map_or(0, |r| r.len()));
                if ih == oh && iw == ow {
                    "same_size"
                } else if oh > ih || ow > iw {
                    "larger_output"
                } else {
                    "smaller_output"
                }
            } else {
                "unknown"
            };

            let entry = per_category.entry(category.to_string()).or_insert((0, 0));
            entry.1 += 1;
            if task_solved {
                entry.0 += 1;
            }

            let status = if task_solved { "SOLVED" } else { "FAILED" };
            let bar = if task_solved { "+" } else { "-" };
            if verbose {
                println!(
                    "  Result: {} ({}/{})\n",
                    status, result.correct, result.total
                );
            } else {
                // Compact output
                let pct = if (idx + 1) > 0 {
                    tasks_solved as f32 / (idx + 1) as f32 * 100.0
                } else {
                    0.0
                };
                print!(
                    "\r  [{:>4}/{}] {} {} | Solved: {}/{} ({:.1}%) | {:.0}ms",
                    idx + 1,
                    tasks_to_eval.len(),
                    bar,
                    task_id,
                    tasks_solved,
                    idx + 1,
                    pct,
                    result.time_ms
                );
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }

        if !verbose {
            println!();
        }

        let total_elapsed = eval_start.elapsed().as_secs_f64() * 1000.0;
        let avg_confidence = if total_tasks > 0 {
            total_confidence / total_tasks as f32
        } else {
            0.0
        };

        // Print summary
        println!("\n================================================================");
        println!("                     EVALUATION RESULTS");
        println!("================================================================");
        println!(
            "  Tasks solved:        {}/{} ({:.1}%)",
            tasks_solved,
            total_tasks,
            if total_tasks > 0 {
                tasks_solved as f64 / total_tasks as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "  Test inputs solved:  {}/{} ({:.1}%)",
            test_inputs_solved,
            total_test_inputs,
            if total_test_inputs > 0 {
                test_inputs_solved as f64 / total_test_inputs as f64 * 100.0
            } else {
                0.0
            }
        );
        println!("  Avg confidence:      {:.4}", avg_confidence);
        println!(
            "  Total time:          {:.1}ms ({:.1}ms/task)",
            total_elapsed,
            if total_tasks > 0 {
                total_elapsed / total_tasks as f64
            } else {
                0.0
            }
        );

        println!("\n  By category:");
        let mut cats: Vec<_> = per_category.iter().collect();
        cats.sort_by_key(|(name, _)| name.clone());
        for (cat, (solved, total)) in &cats {
            println!(
                "    {:<20} {}/{} ({:.1}%)",
                cat,
                solved,
                total,
                if *total > 0 {
                    *solved as f64 / *total as f64 * 100.0
                } else {
                    0.0
                }
            );
        }
        println!("================================================================");

        Ok(EvalSummary {
            total_tasks,
            tasks_solved,
            total_test_inputs,
            test_inputs_solved,
            avg_confidence,
            total_time_ms: total_elapsed,
            per_category,
        })
    }

    // ═══════════════════════════════════════════════════════════════════
    // § 9  CLI
    // ═══════════════════════════════════════════════════════════════════

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();

        let mut data_dir = PathBuf::from("arc-data/data");
        let mut split = "training".to_string();
        let mut single_task: Option<String> = None;
        let mut limit: Option<usize> = None;
        let mut verbose = false;
        let mut max_trials = 2;
        let mut enable_ttt = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--data-dir" => {
                    i += 1;
                    data_dir = PathBuf::from(&args[i]);
                }
                "--split" => {
                    i += 1;
                    split = args[i].clone();
                }
                "--task" => {
                    i += 1;
                    single_task = Some(args[i].clone());
                }
                "--limit" => {
                    i += 1;
                    limit = Some(args[i].parse().expect("limit must be a number"));
                }
                "--verbose" | "-v" => {
                    verbose = true;
                }
                "--trials" => {
                    i += 1;
                    max_trials = args[i].parse().expect("trials must be a number");
                }
                "--ttt" => {
                    enable_ttt = true;
                }
                "--help" | "-h" => {
                    println!("Usage: arc_eval [OPTIONS]");
                    println!();
                    println!("Options:");
                    println!(
                        "  --data-dir <path>  Path to ARC data directory (default: arc-data/data)"
                    );
                    println!("  --split <split>    training | evaluation (default: training)");
                    println!("  --task <id>        Run a single task by ID (e.g. 007bbfb7)");
                    println!("  --limit <n>        Max tasks to evaluate");
                    println!("  --trials <n>       Max trials per test input (default: 2)");
                    println!("  --verbose, -v      Print detailed per-task output");
                    println!("  --ttt              Enable Test-Time Training with LLM fallback");
                    println!("  --help, -h         Show this help");
                    return;
                }
                other => {
                    eprintln!("Unknown argument: {}", other);
                    std::process::exit(1);
                }
            }
            i += 1;
        }

        match evaluate_all(
            &data_dir,
            &split,
            limit,
            single_task.as_deref(),
            max_trials,
            verbose,
            enable_ttt,
        ) {
            Ok(_summary) => {}
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    #[cfg(feature = "layers")]
    arc::run();

    #[cfg(not(feature = "layers"))]
    {
        eprintln!("Error: arc_eval requires the 'layers' feature.");
        eprintln!("Run with: cargo run --features layers --bin arc_eval");
        std::process::exit(1);
    }
}

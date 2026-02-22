// ARC-AGI Evaluation Binary for RustyWorm
// Evaluates the 8-layer multiplicative integration system against ARC-AGI tasks
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
//   --trials <n>         Max trials per test input (default: 3, per ARC rules)

#[cfg(feature = "layers")]
mod arc {
    use rustyworm::mimicry::layers::bridges::BridgeBuilder;
    use rustyworm::mimicry::layers::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
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
        Custom(String),
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

        // Deduplicate
        let mut seen = HashSet::new();
        hypotheses.retain(|t| {
            let key = format!("{:?}", t);
            seen.insert(key)
        });

        hypotheses
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

            ArcSolver { stack }
        }

        pub fn solve(&mut self, task: &ArcTask, max_trials: usize) -> Vec<(Grid, f32)> {
            // 1. Detect candidate transforms
            let candidates = detect_transforms(task);

            // 2. Encode task features into layer confidences
            let confidences = encode_task_to_confidences(task, &candidates);

            // 3. Set difficulty hint based on best candidate confidence
            let best_conf = candidates.first().map(|c| c.confidence).unwrap_or(0.0);
            let difficulty = if best_conf >= 1.0 {
                0.2
            } else if best_conf > 0.5 {
                0.5
            } else {
                0.8
            };
            self.stack.set_difficulty_hint(Some(difficulty));

            // 4. Process through layer system for each test input
            let mut all_results = Vec::new();

            for test_pair in &task.test {
                let mut trial_results: Vec<(Grid, f32)> = Vec::new();

                // Generate candidate outputs for this test input
                for candidate in candidates.iter().take(max_trials.max(3)) {
                    if let Some(output) = apply_transform(&test_pair.input, &candidate.transform) {
                        // Run through layer system to get confidence score
                        let input_state = LayerState::with_confidence(
                            Layer::BasePhysics,
                            confidences.clone(),
                            confidences.base_physics,
                        );

                        self.stack.clear_states();
                        let result = self.stack.process_bidirectional(input_state);

                        // Combined score: transform training accuracy + layer system confidence
                        let layer_conf = result.combined_confidence / 2.0; // normalize from [0,2] to [0,1]
                        let combined_score = candidate.confidence * 0.7 + layer_conf * 0.3;

                        trial_results.push((output, combined_score));
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
    // § 8  EVALUATION HARNESS
    // ═══════════════════════════════════════════════════════════════════

    pub fn evaluate_task(
        solver: &mut ArcSolver,
        task_id: &str,
        task: &ArcTask,
        max_trials: usize,
        verbose: bool,
    ) -> TaskResult {
        let start = Instant::now();

        let predictions = solver.solve(task, max_trials);

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
    ) -> Result<EvalSummary, String> {
        let tasks = load_tasks(data_dir, split)?;
        let mut solver = ArcSolver::new();

        let tasks_to_eval: Vec<_> = if let Some(task_id) = single_task {
            tasks.into_iter().filter(|(id, _)| id == task_id).collect()
        } else if let Some(limit) = limit {
            tasks.into_iter().take(limit).collect()
        } else {
            tasks
        };

        println!("================================================================");
        println!("     RustyWorm ARC-AGI EVALUATION");
        println!("================================================================");
        println!(
            "  Split: {}  |  Tasks: {}  |  Trials: {}",
            split,
            tasks_to_eval.len(),
            max_trials
        );
        println!("  Engine: 8-Layer Multiplicative + Phase 3/5/6 (OCTO Braid)");
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

            let result = evaluate_task(&mut solver, task_id, task, max_trials, verbose);

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
        let mut max_trials = 3;

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
                    println!("  --trials <n>       Max trials per test input (default: 3)");
                    println!("  --verbose, -v      Print detailed per-task output");
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

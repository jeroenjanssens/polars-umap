//! Polars plugin expressions: `umap_fit_transform`, `umap_from_knn` and `knn`.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::Array2;
use nndescent::NNDescent;
use polars::prelude::*;
use polars_arrow::array::{PrimitiveArray, StructArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use umap::{Init, Optimizer, Umap, UmapBuilder};

use crate::convert::{
    embedding_to_series, features_to_matrix, fixed_size_list_from_values, scatter_rows,
    strict_matrix, Matrix, NullPolicy,
};

// ---------------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------------

/// Run `f` on a rayon pool with `n_threads` threads, or on the global pool.
///
/// The plugin links its own copy of rayon, so its pools are separate from the Polars
/// thread pool. The global pool is sized from `RAYON_NUM_THREADS`, else
/// `POLARS_MAX_THREADS`, else the number of logical CPUs. When Polars evaluates many
/// groups concurrently (`.over()`), all of them share that one pool instead of each
/// spawning its own threads.
fn with_pool<R: Send>(n_threads: Option<usize>, f: impl FnOnce() -> R + Send) -> R {
    static GLOBAL_INIT: OnceLock<()> = OnceLock::new();
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

    GLOBAL_INIT.get_or_init(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_none() {
            if let Some(n) = std::env::var("POLARS_MAX_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
            {
                // Fails only if the global pool already exists; that's fine.
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global();
            }
        }
    });

    match n_threads {
        None | Some(0) => f(),
        Some(n) => {
            let pool = {
                let mut pools = POOLS.get_or_init(Default::default).lock().unwrap();
                pools
                    .entry(n)
                    .or_insert_with(|| {
                        Arc::new(
                            rayon::ThreadPoolBuilder::new()
                                .num_threads(n)
                                .thread_name(|i| format!("polars-umap-{i}"))
                                .build()
                                .expect("failed to build thread pool"),
                        )
                    })
                    .clone()
            };
            pool.install(f)
        }
    }
}

// ---------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------

/// Layout parameters shared by `umap_fit_transform` and `umap_from_knn`. Names and
/// defaults follow umap-learn; defaults are filled in on the Python side.
#[derive(Debug, Deserialize)]
pub struct LayoutKwargs {
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    init: String,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    a: Option<f32>,
    b: Option<f32>,
    random_state: Option<u64>,
    optimizer: Option<String>,
    layout: String,
    disconnection_distance: Option<f32>,
    verbose: bool,
    n_threads: Option<usize>,
    gpu: bool,
}

#[derive(Debug, Deserialize)]
pub struct UmapKwargs {
    #[serde(flatten)]
    layout: LayoutKwargs,
    n_neighbors: usize,
    metric: String,
    null_policy: NullPolicy,
    /// When true, the last input is a custom initial embedding.
    custom_init: bool,
}

#[derive(Debug, Deserialize)]
pub struct UmapFromKnnKwargs {
    #[serde(flatten)]
    layout: LayoutKwargs,
    /// When true, the last input is a custom initial embedding.
    custom_init: bool,
}

#[derive(Debug, Deserialize)]
pub struct KnnKwargs {
    k: usize,
    metric: String,
    random_state: Option<u64>,
    null_policy: NullPolicy,
    n_threads: Option<usize>,
    gpu: bool,
    verbose: bool,
}

const METRICS: &[&str] = &[
    "euclidean",
    "l2",
    "manhattan",
    "taxicab",
    "l1",
    "chebyshev",
    "infinity",
    "linfinity",
    "linfty",
    "linf",
    "minkowski",
    "cosine",
    "correlation",
    "hamming",
];

fn check_metric(metric: &str) -> PolarsResult<()> {
    polars_ensure!(
        METRICS.contains(&metric),
        ComputeError: "unknown metric '{}'; expected one of {:?}", metric, METRICS
    );
    Ok(())
}

impl LayoutKwargs {
    fn validate(&self) -> PolarsResult<()> {
        polars_ensure!(self.n_components >= 1, ComputeError: "n_components must be at least 1");
        polars_ensure!(
            self.min_dist >= 0.0 && self.min_dist <= self.spread,
            ComputeError: "min_dist must be in [0, spread], got {} (spread = {})", self.min_dist, self.spread
        );
        polars_ensure!(self.spread > 0.0, ComputeError: "spread must be positive");
        polars_ensure!(self.learning_rate > 0.0, ComputeError: "learning_rate must be positive");
        polars_ensure!(
            (0.0..=1.0).contains(&self.set_op_mix_ratio),
            ComputeError: "set_op_mix_ratio must be in [0, 1]"
        );
        polars_ensure!(
            self.layout == "0.5" || self.layout == "0.6",
            ComputeError: "layout must be \"0.5\" or \"0.6\", got \"{}\"", self.layout
        );
        if let Some(o) = &self.optimizer {
            Optimizer::from_str(o).map_err(|e| polars_err!(ComputeError: "{}", e))?;
        }
        if self.init != "custom" {
            Init::from_str(&self.init).map_err(|e| polars_err!(ComputeError: "{}", e))?;
        }
        Ok(())
    }

    fn init_method(&self, custom: Option<Array2<f32>>) -> Init {
        match custom {
            Some(arr) => Init::Custom(arr),
            None => Init::from_str(&self.init).unwrap(),
        }
    }

    fn apply<'a>(&self, mut b: UmapBuilder<'a>, init: Init) -> UmapBuilder<'a> {
        b = b
            .n_components(self.n_components)
            .min_dist(self.min_dist)
            .spread(self.spread)
            .learning_rate(self.learning_rate)
            .negative_sample_rate(self.negative_sample_rate)
            .repulsion_strength(self.repulsion_strength)
            .local_connectivity(self.local_connectivity)
            .set_op_mix_ratio(self.set_op_mix_ratio)
            .verbose(self.verbose)
            .gpu(self.gpu)
            .compatibility_layout(self.layout == "0.5");
        if let Some(n) = self.n_epochs {
            b = b.n_epochs(n);
        }
        if let Some(a) = self.a {
            b = b.a(a);
        }
        if let Some(v) = self.b {
            b = b.b(v);
        }
        if let Some(seed) = self.random_state {
            b = b.random_state(seed);
        }
        if let Some(d) = self.disconnection_distance {
            b = b.disconnection_distance(d);
        }
        if let Some(o) = &self.optimizer {
            b = b.optimizer(Optimizer::from_str(o).unwrap());
        }
        b.init_method(init)
    }
}

fn umap_err(e: umap::UmapError) -> PolarsError {
    polars_err!(ComputeError: "UMAP failed: {}", e)
}

/// Embedding for tiny inputs, where UMAP is not defined: one point sits at the
/// origin, two points are placed one unit apart along the first axis.
fn trivial_embedding(n: usize, n_components: usize) -> Array2<f32> {
    let mut emb = Array2::zeros((n, n_components));
    if n == 2 {
        emb[[1, 0]] = 1.0;
    }
    emb
}

/// Fit, guarding against non-finite output.
///
/// Upstream's recursive initialization (the default of the 0.6 layout) can produce
/// NaN on some heavily duplicated inputs where umap-learn does not. When that happens
/// with a named init, the fit is retried once with spectral initialization. If the
/// result is still not finite, an error is raised instead of returning NaN.
fn fit_checked<'a, F>(
    lk: &LayoutKwargs,
    custom_init: Option<Array2<f32>>,
    make: F,
) -> PolarsResult<Array2<f32>>
where
    F: Fn() -> UmapBuilder<'a> + Send + Sync,
{
    let is_custom = custom_init.is_some();
    let init = lk.init_method(custom_init);
    let uses_recursive =
        matches!(init, Init::Recursive) || (matches!(init, Init::Auto) && lk.layout == "0.6");
    let emb = with_pool(lk.n_threads, || lk.apply(make(), init).build())
        .map_err(umap_err)?
        .embedding;
    if emb.iter().all(|x| x.is_finite()) {
        return Ok(emb);
    }
    if !is_custom && uses_recursive {
        let emb = with_pool(lk.n_threads, || lk.apply(make(), Init::Spectral).build())
            .map_err(umap_err)?
            .embedding;
        if emb.iter().all(|x| x.is_finite()) {
            return Ok(emb);
        }
    }
    polars_bail!(
        ComputeError:
        "UMAP produced non-finite coordinates; this can happen with heavily duplicated data. \
         Try a different init (e.g. init=\"random\"), deduplicating the rows, or adding a little noise"
    )
}

/// Smallest number of valid rows we hand to the upstream implementation.
const MIN_ROWS: usize = 3;

fn embedding_output(input_fields: &[Field], n_components: usize) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Array(Box::new(DataType::Float32), n_components),
    ))
}

// ---------------------------------------------------------------------------------
// umap_fit_transform
// ---------------------------------------------------------------------------------

fn umap_output(input_fields: &[Field], kwargs: UmapKwargs) -> PolarsResult<Field> {
    embedding_output(input_fields, kwargs.layout.n_components)
}

#[polars_expr(output_type_func_with_kwargs = umap_output)]
fn umap_fit_transform(inputs: &[Series], kwargs: UmapKwargs) -> PolarsResult<Series> {
    let lk = &kwargs.layout;
    lk.validate()?;
    check_metric(&kwargs.metric)?;
    polars_ensure!(kwargs.n_neighbors >= 2, ComputeError: "n_neighbors must be at least 2");

    let (features, init) = if kwargs.custom_init {
        polars_ensure!(inputs.len() >= 2, ComputeError: "custom init requires an init column");
        let (init, features) = inputs.split_last().unwrap();
        (features, Some(init))
    } else {
        (inputs, None)
    };
    let name = features[0].name().clone();
    let mut m = features_to_matrix(features, kwargs.null_policy)?;

    // A custom init is given for all rows; keep only the rows that are fitted.
    let init = match init {
        Some(s) => {
            polars_ensure!(
                s.len() == m.n_rows,
                ComputeError: "init has {} rows but the data has {}", s.len(), m.n_rows
            );
            let s = match &m.valid_rows {
                Some(rows) => {
                    let idx =
                        IdxCa::from_vec("".into(), rows.iter().map(|&r| r as IdxSize).collect());
                    s.take(&idx)?
                }
                None => s.clone(),
            };
            Some(strict_matrix::<f32>(&s, "init", Some(lk.n_components))?)
        }
        None => None,
    };

    let n = m.n_valid();
    let embedding = if n < MIN_ROWS {
        trivial_embedding(n, lk.n_components)
    } else {
        // Like umap-learn: clamp n_neighbors to the number of samples.
        let n_neighbors = kwargs.n_neighbors.min(n - 1);
        let data = Arc::new(std::mem::take(&mut m.data));
        fit_checked(lk, init, || {
            Umap::builder_shared(Arc::clone(&data))
                .n_neighbors(n_neighbors)
                .metric(&kwargs.metric)
        })?
    };
    Ok(embedding_to_series(name, &embedding, &m))
}

// ---------------------------------------------------------------------------------
// umap_from_knn
// ---------------------------------------------------------------------------------

fn umap_from_knn_output(input_fields: &[Field], kwargs: UmapFromKnnKwargs) -> PolarsResult<Field> {
    embedding_output(input_fields, kwargs.layout.n_components)
}

#[polars_expr(output_type_func_with_kwargs = umap_from_knn_output)]
fn umap_from_knn(inputs: &[Series], kwargs: UmapFromKnnKwargs) -> PolarsResult<Series> {
    let lk = &kwargs.layout;
    lk.validate()?;
    polars_ensure!(
        lk.init != "pca",
        ComputeError: "init=\"pca\" needs the original data; it is not available with umap_from_knn"
    );
    let expected = if kwargs.custom_init { 3 } else { 2 };
    polars_ensure!(
        inputs.len() == expected,
        ComputeError: "umap_from_knn expects an indices and a distances column"
    );
    let name = inputs[0].name().clone();
    let n = inputs[0].len();
    polars_ensure!(
        inputs[1].len() == n,
        ComputeError: "indices and distances must have the same number of rows"
    );

    let indices = strict_matrix::<i32>(&inputs[0], "knn indices", None)?;
    let distances = strict_matrix::<f32>(&inputs[1], "knn distances", Some(indices.ncols()))?;
    polars_ensure!(
        indices.iter().all(|&i| i >= 0 && (i as usize) < n),
        ComputeError: "knn indices must be in [0, {})", n
    );
    let init = if kwargs.custom_init {
        Some(strict_matrix::<f32>(
            &inputs[2],
            "init",
            Some(lk.n_components),
        )?)
    } else {
        None
    };

    let m = Matrix {
        data: Array2::zeros((n, 0)),
        n_rows: n,
        valid_rows: None,
    };
    let embedding = if n < MIN_ROWS {
        trivial_embedding(n, lk.n_components)
    } else {
        fit_checked(lk, init, || {
            Umap::builder_from_knn(indices.clone(), distances.clone())
        })?
    };
    Ok(embedding_to_series(name, &embedding, &m))
}

// ---------------------------------------------------------------------------------
// knn
// ---------------------------------------------------------------------------------

fn knn_dtype(k: usize) -> DataType {
    DataType::Struct(vec![
        Field::new("indices".into(), DataType::Array(Box::new(IDX_DTYPE), k)),
        Field::new(
            "distances".into(),
            DataType::Array(Box::new(DataType::Float32), k),
        ),
    ])
}

fn knn_output(input_fields: &[Field], kwargs: KnnKwargs) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        knn_dtype(kwargs.k),
    ))
}

#[polars_expr(output_type_func_with_kwargs = knn_output)]
fn knn(inputs: &[Series], kwargs: KnnKwargs) -> PolarsResult<Series> {
    check_metric(&kwargs.metric)?;
    polars_ensure!(kwargs.k >= 1, ComputeError: "k must be at least 1");
    let k = kwargs.k;
    let name = inputs[0].name().clone();
    let mut m = features_to_matrix(inputs, kwargs.null_policy)?;
    let n = m.n_valid();

    // Row-major (n, k) neighbors in valid-row numbering; -1 marks "no neighbor".
    let (indices, distances): (Vec<i64>, Vec<f32>) = if n == 0 {
        (vec![], vec![])
    } else if n <= k {
        // Fewer points than requested neighbors: brute force over what there is.
        brute_force_knn(&m.data, k, &kwargs.metric)?
    } else {
        let data = Arc::new(std::mem::take(&mut m.data));
        let graph = with_pool(kwargs.n_threads, || {
            let mut b = NNDescent::builder_shared(data, &kwargs.metric, k)
                .verbose(kwargs.verbose)
                .gpu(kwargs.gpu);
            if let Some(seed) = kwargs.random_state {
                b = b.random_state(seed);
            }
            b.build()
        })
        .map_err(|e| polars_err!(ComputeError: "nearest neighbor search failed: {}", e))?
        .neighbor_graph()
        .ok_or_else(|| polars_err!(ComputeError: "nearest neighbor search produced no graph"))?;
        (
            graph.0.iter().map(|&i| i as i64).collect(),
            graph.1.iter().copied().collect(),
        )
    };

    // Map neighbor indices back to original row numbers, nulls for missing ones.
    let idx_values: Vec<IdxSize> = indices
        .iter()
        .map(|&i| {
            if i >= 0 {
                m.original_row(i as usize) as IdxSize
            } else {
                0
            }
        })
        .collect();
    let idx_valid: Vec<bool> = indices.iter().map(|&i| i >= 0).collect();
    let dist_values: Vec<f32> = distances
        .iter()
        .zip(&idx_valid)
        .map(|(&d, &ok)| if ok { d } else { f32::NAN })
        .collect();

    let idx_scattered = scatter_rows(idx_values.into_iter(), k, &m, 0 as IdxSize);
    let valid_scattered = scatter_rows(idx_valid.into_iter(), k, &m, false);
    let dist_scattered = scatter_rows(dist_values.into_iter(), k, &m, f32::NAN);
    let inner_validity = if valid_scattered.iter().all(|v| *v) {
        None
    } else {
        Some(polars_arrow::bitmap::Bitmap::from_iter(valid_scattered))
    };

    let idx_arrow = IDX_DTYPE.to_arrow(CompatLevel::newest());
    let idx_inner = PrimitiveArray::<IdxSize>::from_vec(idx_scattered)
        .to(idx_arrow.clone())
        .with_validity(inner_validity.clone());
    let dist_inner = PrimitiveArray::<f32>::from_vec(dist_scattered).with_validity(inner_validity);

    let outer = m.validity();
    let idx_arr =
        fixed_size_list_from_values(idx_inner.boxed(), k, m.n_rows, outer.clone(), &idx_arrow);
    let dist_arr = fixed_size_list_from_values(
        dist_inner.boxed(),
        k,
        m.n_rows,
        outer.clone(),
        &ArrowDataType::Float32,
    );
    let dtype = knn_dtype(k);
    let arrow_dtype = dtype.to_arrow(CompatLevel::newest());
    let arr = StructArray::new(arrow_dtype, m.n_rows, vec![idx_arr, dist_arr], outer).boxed();
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, vec![arr], &dtype) })
}

fn brute_force_knn(
    data: &Array2<f32>,
    k: usize,
    metric: &str,
) -> PolarsResult<(Vec<i64>, Vec<f32>)> {
    let dist = nndescent::distance::get_distance_func(metric)
        .ok_or_else(|| polars_err!(ComputeError: "unknown metric '{}'", metric))?;
    let n = data.nrows();
    let mut indices = vec![-1i64; n * k];
    let mut distances = vec![f32::NAN; n * k];
    for i in 0..n {
        let mut row: Vec<(f32, usize)> = (0..n)
            .map(|j| {
                (
                    dist(
                        data.row(i).as_slice().unwrap(),
                        data.row(j).as_slice().unwrap(),
                    ),
                    j,
                )
            })
            .collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (slot, (d, j)) in row.into_iter().take(k).enumerate() {
            indices[i * k + slot] = j as i64;
            distances[i * k + slot] = d;
        }
    }
    Ok((indices, distances))
}

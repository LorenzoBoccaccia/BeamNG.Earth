//! A sparse LDL' of a symmetric positive definite matrix on a fill-reducing ordering (AMD), for the staged grade solver's ADMM
//! (server/staged_heights.py, stage 5): the matrix (H + sigma I + A' diag(rho) A) of every sweep is factored once for a rho and
//! solved exactly by two triangular sweeps, the way OSQP's direct solver does it. QDLDL (Stellato et al. 2020, Apache-2.0) and
//! AMD are the ones Clarabel already carries for the shapes' kernel (clarabel::qdldl), pure Rust, the same bits on every platform
//! (no threads, no floating-point reordering): this file only hands them to Python.
use clarabel::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Mutex;

/// The factors of one matrix: SparseLdl(n, indptr, indices, data) takes its upper triangle (column-major CSC, rows sorted in each
/// column, a diagonal entry in every column); refactor(data) puts new values on the same structure (the same ordering) and
/// factors again; solve(b) returns the x of A x = b. An indefinite or singular matrix is an error, never a regularised answer.
#[pyclass(module = "earthfast")]
pub struct SparseLdl {
    f: Mutex<QDLDLFactorisation<f64>>,
    n: usize,
    nnz: usize,
}

fn usize_of(v: &[i64], what: &str) -> PyResult<Vec<usize>> {
    v.iter()
        .map(|&x| usize::try_from(x).map_err(|_| PyValueError::new_err(format!("{what} must not be negative"))))
        .collect()
}

#[pymethods]
impl SparseLdl {
    #[new]
    fn new(
        py: Python<'_>,
        n: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let colptr = usize_of(indptr.as_slice()?, "indptr")?;
        let rowval = usize_of(indices.as_slice()?, "indices")?;
        let nzval = data.as_slice()?.to_vec();
        if colptr.len() != n + 1 || colptr.first() != Some(&0) || colptr.last() != Some(&rowval.len()) || nzval.len() != rowval.len() {
            return Err(PyValueError::new_err("SparseLdl: indptr, indices and data do not describe an n x n CSC matrix"));
        }
        if nzval.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("SparseLdl: the matrix has a value that is not finite"));
        }
        let nnz = nzval.len();
        let a = CscMatrix::new(n, n, colptr, rowval, nzval);
        let settings = QDLDLSettingsBuilder::default()
            .regularize_enable(false)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let f = py
            .detach(move || QDLDLFactorisation::new(&a, Some(settings)))
            .map_err(|e| PyValueError::new_err(format!("SparseLdl: {e}")))?;
        Ok(SparseLdl { f: Mutex::new(f), n, nnz })
    }

    /// New values (the upper triangle's, in the order the matrix was given) on the same structure, factored again.
    fn refactor(&self, py: Python<'_>, data: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let v = data.as_slice()?.to_vec();
        if v.len() != self.nnz || v.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("SparseLdl.refactor: one finite value per stored entry"));
        }
        let idx: Vec<usize> = (0..self.nnz).collect();
        let f = &self.f;
        py.detach(|| {
            let mut g = f.lock().unwrap();
            g.update_values(&idx, &v);
            g.refactor()
        })
        .map_err(|e| PyValueError::new_err(format!("SparseLdl.refactor: {e}")))
    }

    /// The x of A x = b.
    fn solve<'py>(&self, py: Python<'py>, b: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut x = b.as_slice()?.to_vec();
        if x.len() != self.n {
            return Err(PyValueError::new_err("SparseLdl.solve: b is not n long"));
        }
        let f = &self.f;
        py.detach(|| f.lock().unwrap().solve(&mut x));
        Ok(x.into_pyarray(py))
    }

    /// Stored entries of L (the fill the ordering left).
    #[getter]
    fn nnz_l(&self) -> usize {
        self.f.lock().unwrap().nnzL()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseLdl>()?;
    Ok(())
}

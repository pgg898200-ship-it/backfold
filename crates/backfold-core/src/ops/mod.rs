//! Associative statistical and state-space monoids for O(log T) retractable streaming inference.

pub mod covariance;
pub mod cusum;
pub mod eprocess;
pub mod ewma;
pub mod kalman;
pub mod moments;
pub mod ridge;

pub use covariance::{CovarianceMonoid, CovariancePca};
pub use cusum::{CusumMonoid, CusumSide};
pub use eprocess::{ebh_reject, sift_streams_fdr, EProcessMonoid, SiftResult};
pub use ewma::{AffineEwmaMonoid, EwmaMonoid};
pub use kalman::{
    changepoint_bayes_factor_1d, changepoint_bayes_factor_2d, changepoint_bayes_factor_nd,
    changepoint_shock_bayes_factor_1d, changepoint_shock_bayes_factor_2d,
    changepoint_shock_bayes_factor_nd, locate_changepoint_1d, locate_changepoint_2d, smooth_1d,
    smooth_2d, smooth_nd, KalmanFilter1D, KalmanFilter2D, KalmanMonoid, KalmanPosterior,
};
pub use moments::MomentsMonoid;
pub use ridge::{RidgeMonoid, RidgeSolution};

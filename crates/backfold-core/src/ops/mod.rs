//! Associative statistical and state-space monoids for O(log T) retractable streaming inference.

pub mod covariance;
pub mod cusum;
pub mod eprocess;
pub mod ewma;
pub mod kalman;
pub mod moments;
pub mod ridge;

pub use covariance::CovarianceMonoid;
pub use cusum::{CusumMonoid, CusumSide};
pub use eprocess::{ebh_reject, EProcessMonoid};
pub use ewma::{AffineEwmaMonoid, EwmaMonoid};
pub use kalman::{
    smooth_1d, smooth_2d, smooth_nd, KalmanFilter1D, KalmanFilter2D, KalmanMonoid, KalmanPosterior,
};
pub use moments::MomentsMonoid;
pub use ridge::{RidgeMonoid, RidgeSolution};

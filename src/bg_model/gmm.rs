use nalgebra::{Matrix3, Vector3};
use std::f32;

// // PARAMETERS
// TODO: OpenCV MOG2 places bounds on the component variance, see https://github.com/opencv/opencv/blob/7dc88f26f24fa3fd564a282b2438c3ac0263cd2f/modules/video/src/bgfg_gaussmix2.cpp#L105-L113

// // IMPLEMENTATION
// TODO: revisit dependency on nalgebra. This may be overkill and may lead to unnecessary allocations.
// TODO: minimize .unwrap() calls. Code contains many contingencies for possible panics due to extensive use of iterators.
// TODO: float division by zero is well defined to result in 'Inf'. Do we mind?

// // POSSIBLE OPTIMIZATIONS
// TODO: When deciding to remove a component: relying on the sort order of the components list we can take the last one instead of recomputing
// TODO: replace Nalgebra's Vector3 with a simple [f32,3]. Will require implementing simple vector arithmetic methods.
// TODO: use fast (inaccurate floating point) math intrinsics when they reach stable https://doc.rust-lang.org/core/intrinsics/fn.fadd_fast.html
// TODO: clean up redundant calculations of constants like 1-cf
// TODO: constants are currently redundantly stored per pixel... Consider storing only references selected at creation of the model object. Requires specification of lifetime parameters.
// TODO: See weight normalization: is multiplying by the inverse really faster? Benchmark this.

// compute a simplified version of the multivariate normal density, cutting some corners for performance purposes.
fn simplified_mvn_density(x: &Vector3<f32>, mu: &Vector3<f32>, variance: f32) -> f32 {
    // We keep the covariance matrix isotropic (i.e. diagonal and all diagonal elements are equal)
    // this means:
    // - we can compute the determinant simply by taking the product of all diagonal elements
    // - we can compute the inverse simply by taking the inverse of the diagonal entries
    // This drastically simplifies computation.
    let pixel_vector_size = 3;
    //TODO: OPTIMIZE
    let inv_cov = &Matrix3::from_diagonal_element(1.0 / variance);
    let determinant = variance.powi(pixel_vector_size);

    // right hand side of the equation
    let rhs = inv_cov * (x - mu);

    // self.exp() means e^(self)
    // self.tr_dot() returns the dot product between the transpose of self and rhs.
    let top = (-0.5 * (x - mu).transpose().tr_dot(&rhs)).exp();
    let bottom = ((2.0 * f32::consts::PI).powi(pixel_vector_size) * determinant).sqrt();

    return top / bottom;
}

struct GaussianComponent {
    // vector of length 3, representing r,g,b
    mu: Vector3<f32>,

    variance: f32,

    // non-negative, should add up to one across components.
    mixing_weight: f32,
}

impl GaussianComponent {
    // compute the contribution for this component to probability p(x|Xt,BG).
    fn weighted_contribution(&self, xt: &Vector3<f32>) -> f32 {
        let contrib = self.mixing_weight * simplified_mvn_density(xt, &self.mu, self.variance);

        #[cfg(debug_assertions)]
        {
            println!("contrib: {:?}", contrib);
        }

        return contrib;
    }

    // simple multivariate distance of x to the current mu
    fn distance(&self, xt: &Vector3<f32>) -> Vector3<f32> {
        return xt - self.mu;
    }

    // simplified squared mahalanobis distance "D squared"
    fn d_squared(&self, x: &Vector3<f32>) -> f32 {
        let dist = self.distance(x);

        return (dist.tr_dot(&dist.transpose())) / self.variance;
    }

    // update the parameters of the component (note the order)
    fn update_params(&mut self, xt: &Vector3<f32>, owned: bool, alpha: &f32) {
        // update the mixing weight
        let ot = if owned { 1.0 } else { 0.0 };

        self.mixing_weight = self.mixing_weight + alpha * (ot - self.mixing_weight);

        // only update the mu and the variance if the component owns the observation
        if owned {
            let dist = self.distance(xt);
            let damp = alpha / self.mixing_weight;

            // update the mean vector
            self.mu = self.mu + damp * dist;

            // update the variance and apply the constraints
            let var = self.variance + damp * (dist.tr_dot(&dist.transpose()) - self.variance);

            // TODO: make varMAX and varMIN configurable during GMM initialisation.
            // TODO: and find sane defaults (at least ensure nonnegativity)
            // let varMAX = 5.0 * 15.0;
            // let varMIN = 4.0;
            // var = var.max(varMIN);
            // var = var.min(varMAX);
            self.variance = var;
        }

        debug_assert!(self.mixing_weight >= 0.0);
        debug_assert!(self.mu.iter().all(|x| *x >= 0.0));
        debug_assert!(self.variance > 0.0);
    }
}

pub struct GaussianMixtureSettings {
    pub alphamin: f32,
    pub max_components: usize,
    pub initial_variance: f32,
    pub cf: f32,
    pub mahal_threshold: f32, // note: D^2
}

pub struct GaussianMixtureModel {
    // zero or more gaussian components
    components: Vec<GaussianComponent>,

    // maximum alpha decay value envelope
    // aka the learning rate.
    // Higher values will reduce the number of observations needed for new objects to be considered part of the background.
    // As the learning rate needs to be higher at the start of training,
    // new additions to the training data will reduce the alpha until it reaches alphamax.
    // See also cf.
    alphamin: f32,
    alpha: f32,
    training_set_size: u32,

    //the maximum number of components
    max_components: usize,

    // an appropriate initial variance for new components.
    initial_variance: f32,

    // cf is a measure of the maximum portion of the data that can belong to foreground objects
    // without influencing the background model.
    // A new object should be static for log(1 − cf )/ log(1 − α) frames before it is considered part of the background.
    // For example: for cf = 0.1 and α = 0.001 we get 105 frames.
    cf: f32,

    // Distance threshold in terms of squared mahalanobis distance D^2.
    // E.g. if you want a distance threshold of 4 sigma, set this threshold to 4^2.
    mahal_threshold: f32,
}

impl GaussianMixtureModel {
    // debug method to dump some info about the current state of the model
    pub fn summary(&self) -> String {
        let weights: Vec<f32> = self.components.iter().map(|c| c.mixing_weight).collect();
        let variances: Vec<f32> = self.components.iter().map(|c| c.variance).collect();
        let mus: Vec<Vector3<f32>> = self.components.iter().map(|c| c.mu).collect();
        return format!(
            "Components: {}, weights: {:?}, variances: {:?}, mu: {:?}",
            self.components.len(),
            weights,
            variances,
            mus,
        );
    }

    pub fn new(settings: &GaussianMixtureSettings) -> GaussianMixtureModel {
        return GaussianMixtureModel {
            components: Vec::new(),
            alphamin: settings.alphamin,
            alpha: 1.0,
            training_set_size: 0,
            max_components: settings.max_components,
            initial_variance: settings.initial_variance,
            cf: settings.cf,
            mahal_threshold: settings.mahal_threshold,
        };
    }

    pub fn get_heaviest_mean(&self) -> &Vector3<f32> {
        // get the mu of the component with the highest mixing weight.
        // the components are expected to be sorted on descending weight.
        return &self.components[0].mu;
    }

    fn add_new_component(&mut self, xt: &Vector3<f32>) {
        self.components.push(GaussianComponent {
            mu: *xt,
            variance: self.initial_variance,
            mixing_weight: self.alpha,
        })
    }

    // update the model with a new observation.
    pub fn update(&mut self, xt: &Vector3<f32>) {
        // update the learning rate until the minimum is reached
        if self.alpha != self.alphamin {
            self.training_set_size += 1;
            self.alpha = self.alphamin.max(1.0 / self.training_set_size as f32);
        }

        // if there are no components yet, add a new one using this observation.
        if self.components.len() == 0 {
            self.add_new_component(&xt);
            return;
        };

        // compute the mahalanobis distances of the observation to each component
        // and check if any of the distances are within a threshold (i.e. "close" == Dsquared < 3 * variance, aka "within 3 sd")
        let close: Vec<bool> = self
            .components
            .iter()
            .map(|c| return c.d_squared(&xt) < self.mahal_threshold)
            .collect();

        // check if any components are "close" to the observation.
        if close.iter().any(|d| *d) {
            // The "close" component with the highest mixing_weight "owns" the observation. 
            // We update the mu and variance of this component.
            // TODO: Dear gods of clean code, please accept this iterator the way it is. It means well.
            let ((close_and_heaviest, _), _) = self
                .components
                .iter_mut()
                .enumerate()
                .zip(close)
                .filter(|(_, c)| *c)
                .max_by(|((_, comp_a), _), ((_, comp_b), _)| {
                    comp_a
                        .mixing_weight
                        .partial_cmp(&comp_b.mixing_weight)
                        .unwrap()
                }).unwrap();

            // for each component: update the mixing weight
            for (i, c) in self.components.iter_mut().enumerate() {
                if i == close_and_heaviest {
                    c.update_params(xt, true, &self.alpha);
                } else {
                    c.update_params(xt, false, &self.alpha);
                }
            }
        } else {
            // if no components are "close":
            // 1) if the maximum number of components has been reached, discard the component with the smallest weight
            // 2) add a new component using this observation

            if self.components.len() >= self.max_components {
                let (ind, _) = self.components.iter()
                    .enumerate()
                    // Rust requires explicit partial comparisons on floats, see https://github.com/rust-lang/rfcs/issues/1249
                    .min_by(|(_, x), (_,y) | x.mixing_weight.partial_cmp(&y.mixing_weight).unwrap())
                    .unwrap();
                self.components.remove(ind as usize);
            }

            self.add_new_component(&xt);

            // for each component: update the mixing weight (none of them are owners)
            for c in self.components.iter_mut() {
                c.update_params(xt, false, &self.alpha);
            }
        }

        // regardless of what kind of update was performed, re-sort the components on descending weight
        // From the rust docs: When applicable, unstable sorting is preferred because it is generally faster than stable sorting and it doesn't allocate auxiliary memory.
        //  (unstable == may reorder equal elements)
        self.components
            .sort_unstable_by(|a, b| b.mixing_weight.partial_cmp(&a.mixing_weight).unwrap());

        // Renormalize the mixing weights to add up to one like in the OpenCV implementation
        // see https://github.com/opencv/opencv/blob/7dc88f26f24fa3fd564a282b2438c3ac0263cd2f/modules/video/src/bgfg_gaussmix2.cpp#L696-L701
        // Weight normalisation is not that well described in the Zivkovic paper.
        let total_weight = self.components.iter().fold(0.0, |a, b| a + b.mixing_weight);
        let inv_weight = total_weight.recip(); // self.recip() == 1/self
        self.components
            .iter_mut()
            .for_each(|x| x.mixing_weight *= inv_weight);

        debug_assert!(
            self.components
                .iter()
                .all(|c| c.mu.iter().all(|m| m >= &0.0))
        );
    }

    // compute the (BG, FG) probabilities for the xt pixel values.
    pub fn probabilities(&self, xt: &Vector3<f32>) -> (f32, f32) {
        // Select the B largest (highest weight) components using a heuristic.
        // NOTE: we assume that the components vector is already sorted in order of descending mixing weight.
        // This is ensured after every call to the update method.
        let heuristic = 1.0 - self.cf;
        let mut running_weight_sum = 0.0;
        let mut bg_prob = 0.0;
        let mut fg_prob = 0.0;
        for c in self.components.iter() {
            if running_weight_sum < heuristic {
                // Take the sum of the weighted contributions of all components in B, this will make up p(x|Xt, BG).
                bg_prob += c.weighted_contribution(&xt);
            } else {
                fg_prob += c.weighted_contribution(&xt);
            };
            running_weight_sum += c.mixing_weight;
        }

        #[cfg(debug_assertions)]
        {
            if !(bg_prob >= 0.0 && bg_prob <= 1.0) {
                println!("misbehaving bg_prob estimate: {:?}", bg_prob);
                panic!("bg_prob estimate is weird: not between 0 and 1");
            }
        }

        return (bg_prob, fg_prob);
    }
}

#[cfg(test)]
mod tests {

    // use all symbols found in the rest of this file.
    use super::*;

    #[test]
    fn test_multivariate_normal_density() {
        let testdata: Vec<(Vector3<f32>, Vector3<f32>, f32)> = vec![(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 0.0, 0.0),
            // same as multiplying the identity matrix with a variance of 2.5
            2.5,
        )];

        let mpdfs: Vec<f32> = testdata
            .iter()
            .map(|(x, mu, var)| {
                return simplified_mvn_density(x, mu, *var);
            }).collect();

        let expected = vec![0.008815429];

        assert_eq!(expected, mpdfs);
    }

    #[test]
    fn smoke_test_gmm_update() {
        let testdata = get_test_data();
        let mut gmm = get_test_model(&testdata);
        for xt in testdata.iter() {
            gmm.update(&xt)
        }

        // check if the maximum number of components is honored
        assert!(4 >= gmm.components.len());

        // specific to the test data: check if the number of components is equal to k
        assert!(gmm.components.len() == 4);

        // get all the mixing weights
        let weights: Vec<f32> = gmm.components.iter().map(|c| c.mixing_weight).collect();
        println!("mixing weights: {:?}", weights);

        // smoke test the background probability predictions
        let (bg, fg) = gmm.probabilities(&Vector3::new(0.0, 1.0, 300.0));
        println!(
            "background probability: {} \n
            foreground probability: {}",
            bg, fg
        );

        // check if the mixing weights sum up to one
        let summed = weights.iter().fold(0.0, |a, b| a + b);
        assert!(1.0 - summed < 0.00001);
    }

    fn get_test_data() -> Vec<Vector3<f32>> {
        return vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 12.0, 60.0),
            Vector3::new(200.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, 300.0),
            Vector3::new(10.0, 1.0, 300.0),
            Vector3::new(100.0, 1.0, 100.0),
            Vector3::new(10.0, 200.0, 300.0),
        ];
    }

    fn get_test_model(testdata: &Vec<Vector3<f32>>) -> GaussianMixtureModel {
        // This is a pretty insignificant amount of test data. Will not yield anything stable.

        let alphamin = 1.0 / testdata.len() as f32;
        let max_components = 4;
        let cf: f32 = 0.1;

        println!("object needs to be 'static' for {} observations before being considered part of the background",
            (1.0  - cf).log(10.0) / (1.0 - alphamin).log(10.0)
        );

        return GaussianMixtureModel::new(&GaussianMixtureSettings {
            alphamin,
            max_components,
            initial_variance: 20.0,
            cf,
            mahal_threshold: 3.0 * 3.0,
        });
    }

}

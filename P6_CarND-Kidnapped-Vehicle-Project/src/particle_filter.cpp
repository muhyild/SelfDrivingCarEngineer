/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // Set the number of particles
  
  // Create a normal (Gaussian) distribution for x, y and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // initialize all particles based on  gps data
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    
    //Initialize particles and weights vectors
    particle.weight = 1.0;
    particles.push_back(particle); // add the particles in particles vector 
  }  
  // initialization is done
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Create normal distributions for sensor noise
  default_random_engine gen_1;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // Calculate new state based on motion bodel
  for (int i = 0; i < num_particles; i++) {

    if ( fabs(yaw_rate) < 0.0001 ) {
      // when yaw_rate is zero
      particles[i].x += velocity * delta_t * cos( particles[i].theta );
      particles[i].y += velocity * delta_t * sin( particles[i].theta );
      particles[i].theta = particles[i].theta;
    } else {
      // when yaw_rate is not zero
      particles[i].x += velocity / yaw_rate * ( sin( particles[i].theta + yaw_rate * delta_t ) - sin( particles[i].theta ) );
      particles[i].y += velocity / yaw_rate * ( cos( particles[i].theta ) - cos( particles[i].theta + yaw_rate * delta_t ) );
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add random noise 
    particles[i].x += dist_x(gen_1);
    particles[i].y += dist_y(gen_1);
    particles[i].theta += dist_theta(gen_1);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < int(observations.size()); ++i) { // For each observation

    // set minimum distance possible maximum double value
    double min_Distance = numeric_limits<double>::max();

    // Initialize the map th the value that cannot be found in map
    int fnd_ObsvId = -10000;

    for (int j = 0; j < int(predicted.size()); ++j ) { 
      // check each prediction to find the closest (predicted and observed )points using helper functions
      double distance =  dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

      if ( distance <= min_Distance ) {
        min_Distance = distance;
        fnd_ObsvId = predicted[j].id;
      }
    }

    // reassign the id of best predicted position
    observations[i].id = fnd_ObsvId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  //decompose the landmarks into x and y coordinate
  double std_landmarkX = std_landmark[0];
  double std_landmarkY = std_landmark[1];
  // for each paricle 
  for (int i = 0; i < num_particles; i++) {

    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    
    // create a vector to store landmarks
    vector<LandmarkObs> inRangeLandmarks;
    // for each paticle
    for(int j = 0; j < int(map_landmarks.landmark_list.size()); j++) {
      
      float landmark_X = map_landmarks.landmark_list[j].x_f;
      float landmark_Y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // calculate the distance using helper functions
      double distance = dist(x, y, landmark_X, landmark_Y);
      // take the landmarks in sensor range into consideration
      if ( distance<= sensor_range ) {
        inRangeLandmarks.push_back(LandmarkObs{ landmark_id, landmark_X, landmark_Y });
      }
    }

    // transform observation coordinates from vehicle's coordinate system to map coordinates
    vector<LandmarkObs> mappedObservations;
    for(int j = 0; j < int(observations.size()); j++) {
      double XX = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double YY = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      mappedObservations.push_back(LandmarkObs{ observations[j].id, XX, YY });
    }

    // Observation association to landmark
    dataAssociation(inRangeLandmarks, mappedObservations);

    // reset weights
    particles[i].weight = 1.0;
    
    // calculate weights
    for(int j = 0; j < int(mappedObservations.size()); ++j) {
      double observation_X = mappedObservations[j].x;
      double observation_Y = mappedObservations[j].y;

      int landmark_id = mappedObservations[j].id;

      double landmark_X;
      double landmark_Y;
      for (int k = 0; k <int(inRangeLandmarks.size()); ++k){
        if (inRangeLandmarks[k].id == landmark_id){
          landmark_X = inRangeLandmarks[k].x;
          landmark_Y = inRangeLandmarks[k].y;
        }
      }

      // Calculate weights for the observation
      double dX = observation_X - landmark_X;
      double dY = observation_Y - landmark_Y;

      double weight = ( 1/(2*M_PI*std_landmarkX*std_landmarkY)) * exp( -( dX*dX/(2*std_landmarkX*std_landmarkY) + (dY*dY/(2*std_landmarkX*std_landmarkY)) ) );
      
      // weight the observation with total onservation weight 
      if (weight == 0.0) {
        particles[i].weight = 0.0001;
      } else {
        particles[i].weight *= weight;
      }
    }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // create a vector to hold current weights
  vector<double> weights;
  for(int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  } 
  
  default_random_engine gen_2;
  discrete_distribution<size_t> dist_ind(weights.begin(), weights.end());

  vector<Particle> resampled_particles(num_particles);
  
  for (int i = 0; i < num_particles; ++i) {
    int index = dist_ind(gen_2);
    resampled_particles[i] = particles[index];   
  }
  
  // Store the resampled particles
  particles = resampled_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
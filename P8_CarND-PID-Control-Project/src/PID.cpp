#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  tau_p = Kp_;
  tau_i = Ki_;
  tau_d = Kd_;
  
  p_cte = 0;
  d_cte = 0;
  i_cte = 0;
  prev_cte = 0;
  total_cte = 0;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  p_cte = cte; //proportional part
  d_cte = cte - prev_cte; //derivative part
  i_cte = i_cte + cte; // integral part : accumulate the error during run time
  prev_cte = cte; // update the previous error for the derivative part
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  total_cte =  -tau_p * p_cte - tau_d * d_cte - tau_i * i_cte; // calculate the total error
    
  return total_cte;  // TODO: Add your total error calc here!
}
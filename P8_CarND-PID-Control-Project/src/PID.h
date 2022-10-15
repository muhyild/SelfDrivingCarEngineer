#ifndef PID_H
#define PID_H

class PID {
 public:
  
    /**
   * PID Coefficients
   */ 
  double tau_p;
  double tau_i ;
  double tau_d ;
  
   /**
   * PID Errors
   */
  double p_cte;
  double i_cte;
  double d_cte;
  double prev_cte;
  
  /**
  * Parameter Tuning Helper:
  */
  double total_cte;
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();


};

#endif  // PID_H
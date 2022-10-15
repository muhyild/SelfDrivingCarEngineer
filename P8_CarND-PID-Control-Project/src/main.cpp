#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pid;
  /**
   * TODO: Initialize the pid variable.
   */
  // Fine Tuning method will be used: initially all coeefficients are 0
  //pid.Init(0.0,0.0,0.0); // 1 : upss, nothing works :),  check if everything works
  //pid.Init(1.0,0.0,0.0); // 2 : after somewhile the car is out of control, oscillation is big
  //pid.Init(0.5,0.0,0.0); // 3 : better but still oscillates 
  //pid.Init(0.25,0.0,0.0); //4 : better but out of control
  //pid.Init(0.125,0.0,0.0); // 5: better need derivative part
  //pid.Init(0.125,0.0,1.0); // 6: already enought to finish the first round without going outside **** (option 1)
  //pid.Init(0.125,1.0,1.0); // 7: out of control, very sensitive to error, very fast reaction not good
  //pid.Init(0.125,0.5,1.0);  // 8: better but still very bad
  //pid.Init(0.125,0.25,1.0); // 9: no change
  //pid.Init(0.125,0.125,1.0); // 10: still very bad
  //pid.Init(0.125,0.0625,1.0); // 11 : still very bad
  //pid.Init(0.125,0.125,0.0); // 12: derivative part is needed
  //pid.Init(0.125,0.001,1.0);  //  13 : the best so far, a bit oscillation and slow reaction **** (option 2)
  //pid.Init(0.125,0.005,1.0);  // 14 : at the beginning faster, therefore big error, later a bit oscillation
  //pid.Init(0.125,0.0025,1.0); // 15: better 
  //pid.Init(0.125,0.0025,2.0); // 16: better but sometimes oscillates
  //pid.Init(0.25,0.001,3.0); // 17: agressive
  //pid.Init(0.0625,0.001,2.0); // 18: most comfortable, sometimes slow **** (option 3)
  pid.Init(0.0625,0.003,2.0);  // 19. the best so far ******* (option 4, best)
  h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          //double speed = std::stod(j[1]["speed"].get<string>());
          //double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
          pid.UpdateError(cte);
          steer_value = pid.TotalError();
          if (steer_value > 1.0) {
            steer_value = 1.0;
          }
          else if(steer_value < -1.0) {
            steer_value = -1.0;          
          }
            // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value 
                    << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, 
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
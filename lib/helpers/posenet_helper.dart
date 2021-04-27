import 'dart:math';
import 'dart:ui';
import 'package:flutter/services.dart';
import 'package:object_detection/tflite/flat_array.dart';
import 'package:object_detection/tflite/posenet_model.dart';

class PoseNetHelper {

  static List<List<int>> keypointPositions({ FlatArray heatMap }) {
    int numKeypoints = PoseNetModelOutput.keypointSize;
    int height = PoseNetModelOutput.height;
    int width = PoseNetModelOutput.width;
    List<List<int>> keyLocations = [];
    for (int i=0; i<numKeypoints; i++) {
      double maxVal = heatMap.get([0, 0, 0, i]);
      int maxRow = 0;
      int maxCol = 0;
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          if (heatMap.get([0, row, col, i]) > maxVal) {
            maxVal = heatMap.get([0, row, col, i]);
            maxRow = row;
            maxCol = col;
          }
        }
      }
      keyLocations.add([maxRow, maxCol]);
    }
    return keyLocations;
  }

  static double _sigmoid(double value) {
    return (1 / (1 + pow(e, -value)));
  }

  // static keypointScores({ List<List<int>> keyPositions, FlatArray heatMap }) {
  //   return keyPositions.map((e) => _sigmoid(heatMap.get([0, e[0], e[1], e])))
  // }
  
  static List<List<double>> keypointCoordinates({
    List<List<int>> keypointPositions,
    FlatArray offsets
  }) {
    List<List<double>> keypointCoords = [];
    for (int i=0; i<keypointPositions.length; i++) {
      int x = keypointPositions[i][0];
      int y = keypointPositions[i][1];
      double yCoord = y.toDouble() / (PoseNetModelOutput.height - 1).toDouble() * PoseNetModelInput.height.toDouble() + offsets.get([0, y, x, i]);
      double xCoord = x.toDouble() / (PoseNetModelOutput.width - 1).toDouble() * PoseNetModelInput.width.toDouble() + offsets.get([0, y, x, i + PoseNetModelOutput.keypointSize]);
      keypointCoords.add([xCoord, yCoord]);
    }
    return keypointCoords;
  }

  static List<Point> scaledCoords({
    List<List<double>> coords,
    double viewWidth,
    double viewHeight
  }) {
    List<Point> scaled = [];
    for (List<double> coord in coords) {
      double x = coord[0];
      double y = coord[1];
      double xScaled = x * viewWidth / PoseNetModelInput.width;
      double yScaled = y * viewHeight / PoseNetModelInput.height;
      scaled.add(Point(xScaled, yScaled));
    }
    return scaled;
  }
}
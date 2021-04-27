import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/material.dart';

class FlatArray {
  final Tensor tensor;
  final List<double> array;
  final List<int> dimensions;

  FlatArray({ @required this.tensor }) : dimensions = tensor.shape, array = List<double>.from(tensor.data);

  int _flatIndex(List<int> index) {
    if (index.length != dimensions.length) {
      throw ArgumentError('Invalid index: got ${index.length} indices for ${dimensions.length} indices.');
    }
    var offset = 0;
    for (int i=0; i<dimensions.length; i++) {
      if (dimensions[i] <= index[i]) {
        throw ArgumentError('Invalid index: ${index[i]} is bigger than ${dimensions[i]}');
      }
      offset = dimensions[i] * offset + index[i];
    }
    return offset;
  }

  double get(List<int> index) {
    return array[_flatIndex(index)];
  }
}
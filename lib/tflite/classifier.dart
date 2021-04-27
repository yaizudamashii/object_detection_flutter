import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/helpers/posenet_helper.dart';
import 'package:object_detection/tflite/flat_array.dart';
import 'package:object_detection/tflite/posenet_model.dart';
import 'package:object_detection/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as imageLib;
import 'stats.dart';

/// Classifier
class Classifier {
  /// Instance of Interpreter
  Interpreter _interpreter;

  /// Labels file loaded as list
  List<String> _labels;

  // static const String MODEL_FILE_NAME = "detect.tflite";
  static const String MODEL_FILE_NAME = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";
  static const String LABEL_FILE_NAME = "labelmap.txt";

  /// Input size of image (height = width = 300)
  // static const int INPUT_SIZE = 300;
  static const int INPUT_SIZE = PoseNetModelInput.width;

  /// Result score threshold
  static const double THRESHOLD = 0.5;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor imageProcessor;

  /// Padding the image to transform into square
  int padSize;

  /// Shapes of output tensors
  List<List<int>> _outputShapes;
  List<Uint8List> _outputHeats;

  /// Types of output tensors
  List<TfLiteType> _outputTypes;
  List<Uint8List> _outputOffsets;

  /// Number of results to show
  static const int NUM_RESULTS = 10;

  Classifier({
    Interpreter interpreter,
    List<String> labels,
  }) {
    loadModel(interpreter: interpreter);
    // loadLabels(labels: labels);
  }

  /// Loads interpreter from asset
  void loadModel({Interpreter interpreter}) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            MODEL_FILE_NAME,
            // options: InterpreterOptions()..threads = 4,
            options: InterpreterOptions()..threads = Platform.isIOS ? 2 : 4,
          );

      var inputTensors = _interpreter.getInputTensors();
      var outputTensors = _interpreter.getOutputTensors();

      // Tensor heatTensor = outputTensors[0];
      // Tensor offsetsTensor = outputTensors[1];
      // FlatArray heats = FlatArray(tensor: heatTensor);
      // FlatArray offsets = FlatArray(tensor: offsetsTensor);
      //
      // List<List<int>> keypointPositions = PoseNetHelper.keypointPositions(heatMap: heats);
      // List<List<double>> coords = PoseNetHelper.keypointCoordinates(keypointPositions: keypointPositions, offsets:  offsets);
      // List<Point> scaledCoords = PoseNetHelper.scaledCoords(coords: coords);

      _outputShapes = [];
      _outputTypes = [];
      // _outputHeats = [];
      // _outputOffsets = [];
      outputTensors.forEach((tensor) {
        // var dims = tensor.shape;
        // var arr = tensor.data.toList();
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
        // _outputHeats.add(tensor.data);
        // _outputOffsets.add(tensor.data);
      });
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Loads labels from assets
  void loadLabels({List<String> labels}) async {
    try {
      _labels =
          labels ?? await FileUtil.loadLabels("assets/" + LABEL_FILE_NAME);
    } catch (e) {
      print("Error while loading labels: $e");
    }
  }

  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);
    if (imageProcessor == null) {
      imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(padSize, padSize))
          .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeMethod.BILINEAR))
          .build();
    }
    inputImage = imageProcessor.process(inputImage);
    return inputImage;
  }

  /// Runs object detection on the input image
  Map<String, dynamic> predict(imageLib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    // Create TensorImage from image
    TensorImage inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    inputImage = getProcessedImage(inputImage);

    var preProcessElapsedTime = DateTime.now().millisecondsSinceEpoch - preProcessStart;

    // TensorBuffers for output tensors
    TensorBuffer outputLocations = TensorBufferFloat(_outputShapes[0]);
    TensorBuffer outputClasses = TensorBufferFloat(_outputShapes[1]);
    TensorBuffer outputScores = TensorBufferFloat(_outputShapes[2]);
    TensorBuffer numLocations = TensorBufferFloat(_outputShapes[3]);
    // TensorBuffer outputHeats = TensorBufferFloat(_outputHeats[0]);
    // TensorBuffer outputOffsets = TensorBufferFloat(_outputOffsets[0]);

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    // List<Object> inputs = [inputImage.buffer];
    List<Object> inputs = [inputImage.image.getBytes(format: imageLib.Format.bgr)];

    // Outputs map
    Map<int, Object> outputs = {
      0: outputLocations.buffer,
      1: outputClasses.buffer,
      2: outputScores.buffer,
      3: numLocations.buffer,
    };

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    TensorBuffer heatBuffer = TensorBuffer.createFixedSize(<int>[1, 9, 9, 17], TfLiteType.float32);
    TensorBuffer offsetBuffer = TensorBuffer.createFixedSize(<int>[1, 9, 9, 34], TfLiteType.float32);

    // run inference
    _interpreter.runForMultipleInputs(inputs, { 0: heatBuffer, 1: offsetBuffer });

    var inferenceTimeElapsed = DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    // Maximum number of results to show
    int resultsCount = min(NUM_RESULTS, numLocations.getIntValue(0));

    // Using labelOffset = 1 as ??? at index 0
    int labelOffset = 1;

    // Using bounding box utils for easy conversion of tensorbuffer to List<Rect>
    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      valueIndex: [1, 0, 3, 2],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.BOUNDARIES,
      coordinateType: CoordinateType.RATIO,
      height: INPUT_SIZE,
      width: INPUT_SIZE,
    );

    List<Recognition> recognitions = [];

    for (int i = 0; i < resultsCount; i++) {
      // Prediction score
      var score = outputScores.getDoubleValue(i);

      // Label string
      var labelIndex = outputClasses.getIntValue(i) + labelOffset;
      var label = _labels.elementAt(labelIndex);

      if (score > THRESHOLD) {
        // inverse of rect
        // [locations] corresponds to the image size 300 X 300
        // inverseTransformRect transforms it our [inputImage]
        Rect transformedRect = imageProcessor.inverseTransformRect(
            locations[i], image.height, image.width);

        recognitions.add(
          Recognition(i, label, score, transformedRect),
        );
      }
    }

    var predictElapsedTime = DateTime.now().millisecondsSinceEpoch - predictStartTime;

    return {
      "recognitions": recognitions,
      "stats": Stats(
          totalPredictTime: predictElapsedTime,
          inferenceTime: inferenceTimeElapsed,
          preProcessingTime: preProcessElapsedTime)
    };
  }

  /// Gets the interpreter instance
  Interpreter get interpreter => _interpreter;

  /// Gets the loaded labels
  List<String> get labels => _labels;
}

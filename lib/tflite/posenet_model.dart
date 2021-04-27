
class PoseNetModelInput {
  static int batchSize = 1;
  static const int height = 257;
  static const int width = 257;
  static int channelSize = 3;
}

class PoseNetModelOutput {
  static int batchSize = 1;
  static int height = 9;
  static int width = 9;
  static int keypointSize = 17;
  static int offsetSize = 34;
}
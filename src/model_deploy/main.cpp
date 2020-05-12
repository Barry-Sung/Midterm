#include "accelerometer_handler.h"

#include "config.h"

#include "magic_wand_model_data.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"


#include "mbed.h"

#include <cmath>

#include "DA7212.h"

#include "uLCD_4DGL.h"


uLCD_4DGL uLCD(D1, D0, D2);

DA7212 audio;

int16_t waveform[kAudioTxBufferSize];

EventQueue queue(32 * EVENTS_EVENT_SIZE);

Thread t;

int song1[42] = {

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261};


int noteLength1[42] = {

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2};

int song2[32] = {
  261, 294, 330, 261,
  261, 294, 330, 261,
  330, 349, 392,
  330, 349, 392,
  392, 440, 392, 349, 330, 261,
  392, 440, 392, 349, 330, 261,
  261, 392, 261,
  261, 392, 261};

int noteLength2[32] = {
  1, 1, 1, 2,
  1, 1, 1, 2,
  1, 1, 2,
  1, 1, 2,
  1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 2,
  1, 1, 2,
  1, 1, 2};

int song3[24] = {
  392, 330, 330,
  349, 294, 294,
  261, 294, 330, 349, 392, 392, 392,
  392, 330, 330,
  349, 294, 294,
  261, 330, 392, 392, 330};

int noteLength3[24] = {
  1, 1, 2,
  1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2,
  1, 1, 2,
  1, 1, 1, 1, 2};

void playNote(int freq)

{

  for(int i = 0; i < kAudioTxBufferSize; i++)

  {

    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));

  }

  audio.spk.play(waveform, kAudioTxBufferSize);

}

//music
void music1(){
  
  uLCD.printf("\n first : little star \n"); 
  wait(30);
  
  for(int i = 0; i < 42; i++)

  {

    int length1 = noteLength1[i];

    while(length1--)

    {

      // the loop below will play the note for the duration of 1s

      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)

      {

        queue.call(playNote, song1[i]);

      }

      if(length1 < 1) wait(1.0);

    }

  }

}

void music2(){
  
  uLCD.printf("\n sceond : two tigers \n"); 
  wait(30);
  
  for(int i = 0; i < 32; i++)

  {

    int length2 = noteLength2[i];

    while(length2--)

    {

      // the loop below will play the note for the duration of 1s

      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)

      {

        queue.call(playNote, song2[i]);

      }

      if(length2 < 1) wait(1.0);

    }

  }

}

void music3(){
  
  uLCD.printf("\n third : little bee \n"); 
  wait(30);  
  
  for(int i = 0; i < 24; i++)

  {

    int length3 = noteLength3[i];

    while(length3--)

    {

      // the loop below will play the note for the duration of 1s

      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)

      {

        queue.call(playNote, song3[i]);

      }

      if(length3 < 1) wait(1.0);

    }

  }

}




// Return the result of the last prediction

int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }


  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}



int main(int argc, char* argv[]) {


  // Create an area of memory to use for input, output, and intermediate arrays.

  // The size of this will depend on the model you're using, and may need to be

  // determined by experimentation.

  constexpr int kTensorArenaSize = 60 * 1024;

  uint8_t tensor_arena[kTensorArenaSize];


  // Whether we should clear the buffer next time we fetch data

  bool should_clear_buffer = false;

  bool got_data = false;


  // The gesture index of the prediction

  int gesture_index;


  // Set up logging.

  static tflite::MicroErrorReporter micro_error_reporter;

  tflite::ErrorReporter* error_reporter = &micro_error_reporter;


  // Map the model into a usable data structure. This doesn't involve any

  // copying or parsing, it's a very lightweight operation.

  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    error_reporter->Report(

        "Model provided is schema version %d not equal "

        "to supported version %d.",

        model->version(), TFLITE_SCHEMA_VERSION);

    return -1;

  }


  // Pull in only the operation implementations we need.

  // This relies on a complete list of all the ops needed by this graph.

  // An easier approach is to just use the AllOpsResolver, but this will

  // incur some penalty in code space for op implementations that are not

  // needed by this graph.

  static tflite::MicroOpResolver<5> micro_op_resolver;

  micro_op_resolver.AddBuiltin(

      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());
  
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               
                               tflite::ops::micro::Register_RESHAPE(), 1);


  // Build an interpreter to run the model with

  static tflite::MicroInterpreter static_interpreter(

      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  tflite::MicroInterpreter* interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors

  interpreter->AllocateTensors();


  // Obtain pointer to the model's input tensor

  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||

      (model_input->dims->data[1] != config.seq_length) ||

      (model_input->dims->data[2] != kChannelNumber) ||

      (model_input->type != kTfLiteFloat32)) {

    error_reporter->Report("Bad input tensor parameters in model");

    return -1;

  }


  int input_length = model_input->bytes / sizeof(float);


  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {

    error_reporter->Report("Set up failed\n");

    return -1;

  }


  error_reporter->Report("Set up successful...\n");


  while (true) {


    // Attempt to read new data from the accelerometer

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);


    // If there was no new data,

    // don't try to clear the buffer again and wait until next time

    if (!got_data) {

      should_clear_buffer = false;

      continue;

    }


    // Run inference, and report any error

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {

      error_reporter->Report("Invoke failed on index: %d\n", begin_index);

      continue;

    }


    // Analyze the results to obtain a prediction

    gesture_index = PredictGesture(interpreter->output(0)->data.f);


    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;


    // Produce an output

    if (gesture_index < label_num) {
      
      t.start(callback(&queue, &EventQueue::dispatch_forever));
      
      if(gesture_index = 0){  
        error_reporter->Report(config.output_message[gesture_index]);
        music1();
      }
      if(gesture_index = 1){  
        error_reporter->Report(config.output_message[gesture_index]);
        music2();
      }
      if(gesture_index = 0){  
        error_reporter->Report(config.output_message[gesture_index]);
        music3();
      }
    }

  }

}

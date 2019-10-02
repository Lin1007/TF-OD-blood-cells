This project is an example to use tensorflow object detection in colaboratory

For more informations, consult [tensorflow object detection docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) or [tensorflow object detection GitHub](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Install

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#

# Training custom object detector

1. Create dataset or download dataset. 

2. Label images(For example using labelImages), save it as pascal voc

3. Convert xml to csv usint `xml2csv.py`

   ```shell
   # Create train data:
   python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv
   
   # Create test data:
   python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
   ```

4. Generate tfrecords

   ```shell
   # Create train data:
   python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record
   
   # Create test data:
   python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
   ```

5. Generate label maps:

   ```json
   item {
   	id: 1
   	name : 'RBC'
   }
   
   item {
   	id:2
   	name : 'WBC'
   }
   item {
   	id:3
   	name : 'Platelets'
   }
   ```

   

6. Download pertained model [model list](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) are showed in github, which also specify the speed and meam average precision.

7. Make a copy of the config file `model_name.config`

   - Set minimum dimension: the image will be resized to has at least this dimension
   - fine_tune_checkpoint: the pretrained model
   - tfrecord files train and test: input files
   - label_map_path
   - Training batch_size: increase/decrease depending on the available memory. Higher value require more memory
   - Training steps
   - Number of classes: set this to the number of different label classes
   - Number of validation examples.

   ```json
   model {
     ssd {
       num_classes: 3
       box_coder {
         faster_rcnn_box_coder {
           y_scale: 10.0
           x_scale: 10.0
           height_scale: 5.0
           width_scale: 5.0
         }
       }
       matcher {
         argmax_matcher {
           matched_threshold: 0.5
           unmatched_threshold: 0.5
           ignore_thresholds: false
           negatives_lower_than_unmatched: true
           force_match_for_each_row: true
         }
       }
       similarity_calculator {
         iou_similarity {
         }
       }
       anchor_generator {
         ssd_anchor_generator {
           num_layers: 6
           min_scale: 0.2
           max_scale: 0.95
           aspect_ratios: 1.0
           aspect_ratios: 2.0
           aspect_ratios: 0.5
           aspect_ratios: 3.0
           aspect_ratios: 0.3333
         }
       }
       image_resizer {
         fixed_shape_resizer {
           height: 300
           width: 300
         }
       }
       box_predictor {
         convolutional_box_predictor {
           min_depth: 0
           max_depth: 0
           num_layers_before_predictor: 0
           use_dropout: false
           dropout_keep_probability: 0.8
           kernel_size: 1
           box_code_size: 4
           apply_sigmoid_to_scores: false
           conv_hyperparams {
             activation: RELU_6,
             regularizer {
               l2_regularizer {
                 weight: 0.00004
               }
             }
             initializer {
               truncated_normal_initializer {
                 stddev: 0.03
                 mean: 0.0
               }
             }
             batch_norm {
               train: true,
               scale: true,
               center: true,
               decay: 0.9997,
               epsilon: 0.001,
             }
           }
         }
       }
       feature_extractor {
         type: 'ssd_mobilenet_v2'
         min_depth: 16
         depth_multiplier: 1.0
         conv_hyperparams {
           activation: RELU_6,
           regularizer {
             l2_regularizer {
               weight: 0.00004
             }
           }
           initializer {
             truncated_normal_initializer {
               stddev: 0.03
               mean: 0.0
             }
           }
           batch_norm {
             train: true,
             scale: true,
             center: true,
             decay: 0.9997,
             epsilon: 0.001,
           }
         }
       }
       loss {
         classification_loss {
           weighted_sigmoid {
           }
         }
         localization_loss {
           weighted_smooth_l1 {
           }
         }
         hard_example_miner {
           num_hard_examples: 3000
           iou_threshold: 0.99
           loss_type: CLASSIFICATION
           max_negatives_per_positive: 3
           min_negatives_per_image: 3
         }
         classification_weight: 1.0
         localization_weight: 1.0
       }
       normalize_loss_by_num_matches: true
       post_processing {
         batch_non_max_suppression {
           score_threshold: 1e-8
           iou_threshold: 0.6
           max_detections_per_class: 100
           max_total_detections: 100
         }
         score_converter: SIGMOID
       }
     }
   }
   
   train_config: {
     batch_size: 12
     optimizer {
       rms_prop_optimizer: {
         learning_rate: {
           exponential_decay_learning_rate {
             initial_learning_rate: 0.004
             decay_steps: 800720
             decay_factor: 0.95
           }
         }
         momentum_optimizer_value: 0.9
         decay: 0.9
         epsilon: 1.0
       }
     }
     fine_tune_checkpoint: "/content/models/research/pretrained_model/model.ckpt"
     fine_tune_checkpoint_type:  "detection"
     # Note: The below line limits the training process to 200K steps, which we
     # empirically found to be sufficient enough to train the pets dataset. This
     # effectively bypasses the learning rate schedule (the learning rate will
     # never decay). Remove the below line to train indefinitely.
     num_steps: 5000
     data_augmentation_options {
       random_horizontal_flip {
       }
     }
     data_augmentation_options {
       ssd_random_crop {
       }
     }
   }
   
   train_input_reader: {
     tf_record_input_reader {
       input_path: "/content/TF-OD-blood-cells/BC_dataset/train.record"
     }
     label_map_path: "/content/TF-OD-blood-cells/BC_dataset/label_map.pbtxt"
   }
   
   eval_config: {
     num_examples: 935
     # Note: The below line limits the evaluation process to 10 evaluations.
     # Remove the below line to evaluate indefinitely.
     max_evals: 10
   }
   
   eval_input_reader: {
     tf_record_input_reader {
       input_path: "/content/TF-OD-blood-cells/BC_dataset/test.record"
     }
     label_map_path: "/content/TF-OD-blood-cells/BC_dataset/label_map.pbtxt"
     shuffle: false
     num_readers: 1
   }
   ```

   

8. Train the model

   ```shell
   pipeline_fname={path to pipeline config file}
   model_dir={path to model directory}
   num_training_steps=50000
   num_eval_steps = 200
   !python /content/models/research/object_detection/model_main.py \
       --pipeline_config_path={pipeline_fname} \
       --model_dir={model_dir} \
       --alsologtostderr \
       --num_train_steps={num_training_steps} \
       --num_eval_steps={num_eval_steps}
   ```

   

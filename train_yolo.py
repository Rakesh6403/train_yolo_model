from ultralytics import YOLO

# .yaml file
# train: /Home/RR/ESRGAN/set
# val: /Home/RR/ESRGAN/set
# nc: 9
# names: ['A-vehicle', 'armed', 'plz-05', 'Fortification', 'Arty', 'iveco', 'Hy', 'LTV', 'guilly_suit']

if __name__ == '__main__':
    # Load a YOLOv8n model pre-trained on COCO dataset
    model = YOLO('yolov8m.pt')  # or 'yolov8n.yaml' for a fresh model

    # Train the model
    results = model.train(
        data='set/data.yaml',  # Path to dataset configuration file
        epochs=120,  # Number of epochs to train for
        imgsz=640,  # Image size
        batch=16,  # Batch size
        workers=8,  # Number of dataloader workers
        device=0,  # GPU device to use, 'cpu' for CPU
        name='yolov8n_custom',  # Name of the experiment
        pretrained=True,  # Whether to use a pretrained model
        verbose=True  # Print training progress
    )

    # # Save the trained model
    # model.save('yolov8n_custom_best.pt')  # Save the best model to a specific file

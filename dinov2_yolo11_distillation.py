import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 ViT-B/14 model.
    lightly_train.train(
        out="out/dinov2_domain_adapted2",
        data="image_data/rivendale/firefly_left",
        model="dinov2/vitl14",
        method="dinov2",
        batch_size=4,
        # resume_interrupted=True
        # Add pretrained checkpoint for DINOv2
        transform_args={"image_size": (1088, 1440)}
    )

    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.train(
        out="out/yolo11_distillation",
        data="datasets/rivendale_v5/images/train",
        model="ultralytics/yolov8l.pt",
        method="distillation",
        method_args={
            "teacher": "dinov2/vitl14",
            "teacher_weights": "out/dinov2_domain_adapted2/exported_models/exported_last.pt", # pretrained `dinov2/vitl14` weights
        },
        batch_size=16,
    )
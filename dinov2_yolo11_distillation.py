import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 ViT-B/14 model.
    lightly_train.train(
        out="out/dinov2_domain_adapted",
        data="datasets/firefly_left_all",
        model="dinov2/vitl14",
        method="dinov2",
        # batch_size=4,
        # resume_interrupted=True
        transform_args={"image_size": (1088, 1440)}
    )

    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.train(
        out="out/yolo11_distillation",
        data="datasets/firefly_left_all",
        model="ultralytics/yolov8l.pt",
        method="distillation",
        method_args={
            "teacher": "dinov2/vitl14",
            "teacher_weights": "out/dinov2_domain_adapted/exported_models/exported_last.pt", # pretrained `dinov2/vitl14` weights
        },
        # batch_size=4,
        transform_args={"image_size": (1088, 1440)}
    )
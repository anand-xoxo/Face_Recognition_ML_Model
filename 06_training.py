#Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


#Saving the model
model.save('/content/drive/MyDrive/practice_hehe/face_recognition_model.h5')
print("Model saved to Google Drive.")

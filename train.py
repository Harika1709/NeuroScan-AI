import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, accuracy_score

# ================== DATA ==================
train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_data = train.flow_from_directory(
    'dataset/Training',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train.flow_from_directory(
    'dataset/Training',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_data.class_indices)

# ================== BASE MODEL ==================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base layers initially
for layer in base_model.layers:
    layer.trainable = False

# ================== CUSTOM HEAD ==================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ================== COMPILE ==================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================== CALLBACKS ==================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.3)

# ================== TRAIN ==================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# ================== FINE-TUNING ==================
# Unfreeze top layers for better accuracy
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ================== SAVE ==================
model.save("best_model.keras")

# ================== EVALUATION ==================
val_data.reset()
pred = model.predict(val_data)

pred_classes = pred.argmax(axis=1)
true_classes = val_data.classes

print("\n===== FINAL METRICS =====")

acc = accuracy_score(true_classes, pred_classes)
print("Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(
    true_classes,
    pred_classes,
    target_names=list(val_data.class_indices.keys())
))
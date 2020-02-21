import collections
import numpy as np
import tensorflow as tf

from models import large_model, small_model

def softmax_with_temp(logits, temp=1):
    logits = (logits - tf.math.reduce_max(logits)) / temp
    exp_logits = tf.math.exp(logits)
    logits_sum = tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True)
    result = exp_logits / logits_sum
    return result

def custom_ce(y_true, y_soft, y_pred, y_soft_pred, alpha = 0.5):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    origin_loss = -tf.math.reduce_mean(tf.math.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=False))

    y_soft = tf.clip_by_value(y_soft, 1e-7, 1 - 1e-7)
    y_soft_pred = tf.clip_by_value(y_soft_pred, 1e-7, 1 - 1e-7)
    soft_loss = -tf.math.reduce_mean(tf.math.reduce_sum(y_soft * tf.math.log(y_soft_pred), axis=-1, keepdims=False))
    
    return alpha * soft_loss + (1 - alpha) * origin_loss


if __name__ == "__main__":
    temp = 3
    batch_sz = 100
    epochs = 10000

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')  
    train_images /= 255
    test_images /= 255

    train_images = train_images.reshape((len(train_images), -1))
    test_images = test_images.reshape((len(test_images), -1))

    train_labels = tf.keras.utils.to_categorical(train_labels.astype('float32'))
    test_labels = tf.keras.utils.to_categorical(test_labels.astype('float32'))

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.repeat(epochs).batch(batch_sz)

    ## Load Teacher model
    with open('model_config.json') as json_file:
        json_config = json_file.read()
    teacher_model = tf.keras.models.model_from_json(json_config)
    teacher_model.load_weights('model_weights.h5')
    
    ### remove softmax in Teacher model
    teacher_model_ex_softmax = tf.keras.Model(inputs=teacher_model.input, outputs=teacher_model.get_layer('logits').output)

    ## make student model
    student_model = small_model(train_images.shape[-1], 10)
    logits = student_model.get_layer('logits').output
    
    student_model = tf.keras.Model(inputs=student_model.input, outputs=logits)
    student_model.summary()

    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    # Define our metrics
    train_acc = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
    test_acc = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    train_log_dir = 'kd/train'
    test_log_dir = 'kd/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    def train_step(images, labels):
        with tf.GradientTape() as tape:
            pred = student_model(images, training=True)

            unsoft_pred = softmax_with_temp(pred, 1)
            soft_pred = softmax_with_temp(pred, temp)

            teacher_logits = teacher_model_ex_softmax(images)
            softened_teacher_prob = softmax_with_temp(teacher_logits, temp)

            loss_value = custom_ce(labels, softened_teacher_prob, unsoft_pred, soft_pred)

        grads = tape.gradient(loss_value, student_model.trainable_variables)
        opt.apply_gradients(zip(grads, student_model.trainable_variables))

        train_acc(labels, pred)
        train_loss(loss_value)

        return loss_value

    @tf.function
    def train():

        step = 0
        ckpt_step = 0
        ckpt_step = tf.cast(ckpt_step, tf.int64)

        for x, y in dataset:
            loss = train_step(x, y)
            
            test_acc(test_labels, student_model(test_images, training=False))

            step += 1

            if step % 100 == 0:
                tf.print(step, int(len(train_images) * epochs / batch_sz), train_loss.result(), train_acc.result(), test_acc.result())

            if step % int(len(train_images) / batch_sz) == 0:
                ckpt_step += 1
                with train_summary_writer.as_default():
                    tf.summary.scalar('epoch_accuracy', train_acc.result(), step=ckpt_step)
                    tf.summary.scalar('epoch_loss', train_loss.result(), step=ckpt_step)
                with test_summary_writer.as_default():
                    tf.summary.scalar('epoch_accuracy', test_acc.result(), step=ckpt_step)
                
                train_acc.reset_states()
                test_acc.reset_states()

                train_loss.reset_states()

    train()
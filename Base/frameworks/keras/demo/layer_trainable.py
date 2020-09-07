

model = ...


for layer in model.layers[:-8]:
    layer.trainable = False
print(model.summary())

model.compile(loss='binary_crossentropy',  
            optimizer=opt, 
            metrics=[binary_accuracy])#fmeasure

model.fit_generator(myGenerator(train_generator,cate_names_final,pre_to_label), 
    validation_data=myGenerator(val_generator,cate_names_final,pre_to_label), 
    steps_per_epoch=count_train // batch_size,
    validation_steps=count_val // batch_size,
    epochs=cfg['epochs'],
    class_weight='auto',
    callbacks=[earlystop,checkpoint,reduce_lr])



for layer in model.layers[:-8]:
    layer.trainable = True
print(model.summary())

model.compile(loss='binary_crossentropy',  
            optimizer=opt, 
            metrics=[binary_accuracy])#fmeasure

model.fit_generator(myGenerator(train_generator,cate_names_final,pre_to_label), 
    validation_data=myGenerator(val_generator,cate_names_final,pre_to_label), 
    steps_per_epoch=count_train // batch_size,
    validation_steps=count_val // batch_size,
    epochs=cfg['epochs'],
    class_weight='auto',
    callbacks=[earlystop,checkpoint,reduce_lr])

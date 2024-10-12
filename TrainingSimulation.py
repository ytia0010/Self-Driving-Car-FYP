#####import library
from utlis import *


#### STEP 1
path = 'SimulationData'
data = importDataInfo(path)

 #### STEP 2
data = balanceData(data,display=True)

#### STEP 3
# imagesPath, steerings = loadData(path,data)
#
# #### STEP 4
# xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
# print('Total Training Images: ',len(xTrain))
# print('Total Validation Images: ',len(xVal))
#
#
# ######
# model = createModel()
# model.summary()
#
# history = model.fit(batchGen(xTrain, yTrain, 100, 1),
#                                   steps_per_epoch=300,
#                                   epochs=10,
#                                   validation_data=batchGen(xVal, yVal, 100, 0),
#                                   validation_steps=200)
#
# model.save('model.h5')
# print('Model Saved')
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training', 'Validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.show()
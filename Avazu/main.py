from logistic_regression import logistic_regression
from util import readData,log_loss,readData_ffm
from fm import fm_regression
from field_aware_fm import ffm_regression

# log_learner = logistic_regression(0.01, 0. ** 5, 2 ** 20, method='None')
# count = 1
# loss_count = 0
# loss = 0
# for ID, x, y in readData('datasets/train.csv', 2 ** 20):
#     p = log_learner.predict(x)
#
#     if count % 100000 == 0:
#         loss_count +=1
#         loss += log_loss(p, y)
#     else:
#         log_learner.update(x, p, y)
#
#     count += 1
#
# print(loss/loss_count)
#
# with open('datasets/submission_lr.csv', 'w') as outfile:
#     outfile.write('id,click\n')
#     for ID, x, y in readData('datasets/test.csv', 2**20):
#         p = log_learner.predict(x)
#         outfile.write('%s,%s\n' % (ID, str(p)))
#
#
# fm_learner = fm_regression(0.01,0,2**20)
# count = 1
# loss_count = 0
# loss = 0
# for ID,x, y in readData('datasets/train.csv', 2 ** 20):
#     p = fm_learner.predict(x)
#
#     if count % 100000 == 0:
#         loss_count +=1
#         loss += log_loss(p, y)
#     else:
#         fm_learner.update(x, p, y)
#
#     count += 1
#
# print(loss/loss_count)
#
# with open('datasets/submission_fm.csv', 'w') as outfile:
#     outfile.write('id,click\n')
#     for ID, x, y in readData('datasets/test.csv', 2**20):
#         p = fm_learner.predict(x)
#         outfile.write('%s,%s\n' % (ID, str(p)))

# log_learner = logistic_regression(0.01, 0.1 ** 4, 2 ** 20, method='None')
# count = 1
# loss_count = 0
# loss = 0
# for ID, x, y in readData('datasets/train.csv', 2 ** 20):
#     p = log_learner.predict(x)
#
#     if count % 7 == 0:
#         loss_count += 1
#         loss += log_loss(p, y)
#     else:
#         log_learner.update(x, p, y)
#
#     count += 1
#
#     if count == 100001:
#         break
#
# print(loss/loss_count)

# with open('datasets/submission_lr_L2.csv', 'w') as outfile:
#     outfile.write('id,click\n')
#     for ID, x, y in readData('datasets/test.csv', 2**20):
#         p = log_learner.predict(x)
#         outfile.write('%s,%s\n' % (ID, str(p)))

ffm_learner = ffm_regression(0.01,0.1**4,2**20)
count = 1
loss_count = 0
loss = 0
for ID,x, y in readData_ffm('datasets/train.csv', 2 ** 20):
    p = ffm_learner.predict(x)

    if count % 100000 == 0:
        loss_count += 1
        loss += log_loss(p, y)
    else:
        ffm_learner.update(x, p, y)

    count += 1

print(loss/loss_count)

with open('datasets/submission_ffm.csv', 'w') as outfile:
    outfile.write('id,click\n')
    for ID, x, y in readData_ffm('datasets/test.csv', 2**20):
        p = ffm_learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
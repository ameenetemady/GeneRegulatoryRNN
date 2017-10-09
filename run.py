import grrnnLib as grrnnLib

# 0) Data settings
strBaseDir = 'data/app18_net9s/'
dicTrainSettings = {'KO': '{!s}/KO.tsv'.format(strBaseDir),
                    'TFs': '{!s}/TFs.tsv'.format(strBaseDir),
                    'NonTFs': '{!s}/NonTFs.tsv'.format(strBaseDir),
                    'Graph': '{!s}/Graph.tsv'.format(strBaseDir)}

dicTestSettings = {'KO': '{!s}/test_KO.tsv'.format(strBaseDir),
                   'TFs': '{!s}/test_TFs.tsv'.format(strBaseDir),
                   'NonTFsPred': '{!s}/test_NonTFsPred.tsv'.format(strBaseDir),
                   'Graph': '{!s}/Graph.tsv'.format(strBaseDir)}                

# 1) Train
model = grrnnLib.createModel(dicTrainSettings['Graph'])
dataTrain = grrnnLib.loadData(dicTrainSettings)
model.train(dataTrain.input, dataTrain.target)

# 2) Test
dataTest = grrnnLib.loadData(dicTestSettings)
predNonTFsTest = model.predict(dataTest.input, dataTest.target)
grrnnLib.save(predNonTFsTest)
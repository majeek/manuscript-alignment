from os import listdir
from os import path


def create_pairing_information(root_path):
    subWordMap = {}
    subWordIndex = 0
    for firstBook in books:
        for secondBook in books:
            if books.index(firstBook) >= books.index(secondBook):
                continue
            print('pairing ' + firstBook + ' with ' + secondBook + ' ...')
            firstBookDirs = [d for d in listdir(root_path + '/' + firstBook + '/')]
            secondBookDirs = [d for d in listdir(root_path + '/' + secondBook + '/')]

            i = 0
            for fbDir in firstBookDirs:
                print(i, 'of', len(firstBookDirs) - 1)
                with open(root_path + firstBook + '/' + fbDir + '/' + fbDir + '.txt', 'r') as fbFile:
                    fImages = []
                    for line in fbFile.readlines():
                        line = line.decode('utf-8-sig').strip()
                        imgName, label = line.split('\t')
                        fImages.append((imgName, label.strip()))

                    for sbDir in secondBookDirs:
                        pairs = []
                        labels = []
                        with open(root_path + secondBook + '/' + sbDir + '/' + sbDir + '.txt', 'r') as sbFile:
                            sImages = []
                            for sLine in sbFile.readlines():
                                sLine = sLine.decode('utf-8-sig').strip()
                                sImgName, sLabel = sLine.split('\t')
                                sImages.append((sImgName, sLabel.strip()))
                            for fImg in fImages:
                                for sImg in sImages:
                                    if fImg[1] == sImg[1]:
                                        if fImg[1] in subWordMap:
                                            labels.append(subWordMap[fImg[1]])
                                        else:
                                            subWordMap[fImg[1]] = subWordIndex
                                            labels.append(subWordIndex)
                                            subWordIndex += 1
                                        pairs.append((firstBook + '-' + fbDir + '-' + fImg[0] + '.png',
                                                      secondBook + '-' + sbDir + '-' + sImg[0] + '.png',
                                                      subWordMap[fImg[1]]
                                                      ))
                        with open(path.join(root_path + 'pairs/',
                                            firstBook + '-' + fbDir + '+' + secondBook + '-' + sbDir + '.txt'),
                                  'w+') as fp:
                            fp.write('\n'.join('%s\t%s\t%s' % x for x in pairs))
                            fp.write('\n')
                i += 1


books = ['0207', '0206']
root = '/DATA/majeek/data/0206-0207/combined/'
print('training information..')
create_pairing_information(root + 'train/')
print('testing and validation information..')
create_pairing_information(root + 'test_valid/')
print('done')
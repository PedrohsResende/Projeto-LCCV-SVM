import matplotlib.pyplot as plt #plotar gráficos
from sklearn import datasets, svm, metrics  #a própria svm para o dataset Digits
import pickle #salvar no disco e usar depois

def trainSVM():
    #carregar base de dados
    digits = datasets.load_digits()

    #transformar as imagens 8x8 em vetor de 64 posições
    numberOFSamples = len(digits.images)
    data = digits.images.reshape((numberOFSamples, -1))
    print(data.shape) # número de 1797 imagens 8x8

    #criar o classificador
    classificador = svm.SVC(gamma = "scale")

    #treinar o clasificador com 50% das imagens na base de dados
    classificador.fit(data[:numberOFSamples // 2 ], digits.target[:numberOFSamples // 2])

    #salvar no disco para não precisar treinar a cada teste
    with open('aprendizado.joblib', 'wb') as PonteiroDisco:
        pickle.dump(classificador, PonteiroDisco)


def loadSVM():
    #carregar base de dados
    digits = datasets.load_digits()

    #transformar as imagens 8x8 em vetor de 64 posições
    numberOFSamples = len(digits.images)
    data = digits.images.reshape((numberOFSamples, -1))

    #carregar no disco o modelo já treiando
    with open('aprendizado.joblib', 'rb') as PonteiroDisco:
        classificador = pickle.load(PonteiroDisco)

    #predizer os rótulos da segunda metade das imagens no banco de dados
    rotulosEsperados = digits.target[numberOFSamples // 2 :]
    rotulosPreditos = classificador.predict(data[numberOFSamples // 2 :])

    #matrizes de confusão
    print(classificador, metrics.classification_report(rotulosEsperados, rotulosPreditos))
    print(metrics.confusion_matrix(rotulosEsperados, rotulosPreditos))

    #exibir 12 imagens do dataset com as classificações
    listaImagensPredicoes = list(zip(digits.images[numberOFSamples // 2 :], rotulosPreditos))
    for index, (images, predicao) in enumerate(listaImagensPredicoes[:12:]):
        print(index)
        plt.subplot(3, 4, index + 1)
        plt.axis('off')
        plt.imshow(images, interpolation='nearest')
        plt.title("Predicao: %i" % predicao)

    plt.show()
        
if __name__ == "__main__":
    #trainSVM()
    loadSVM()
import fire, os, json
import matplotlib.pyplot as plt

'''
plots roc curves reading all the data from a directory
'''
def main(root='experiments', filterNames={'ImdbTwitterBernoulliBOW(1,3)', 'TwitterImdbBernoulliBOW(1,3)'}):
    curves = []
    files = os.listdir(root)
    for f in files:
        path = root+'/'+f
        if os.path.isfile(path):
            fp = open(path)
            d = json.load(fp)
            fp.close()
            if filterNames is None or d['name'] in filterNames:
                curves.append((d['name'], d['tpr'], d['fpr']))

    for c in curves:
        plt.plot(c[2], c[1], label=c[0])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    fire.Fire(main)
from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier, RandomForestClassifier
from boosted_decision_tree import HGBClassifier
from pickle import dump, load

sys.setrecursionlimit(10000)

def train_test_BDT(train, val, test, sigma, full_supervision=False, n_folds=50):        
        
        pred_list_all = []
        #kf = KFold(n_splits = 10)
        train = np.vstack((train,val))
        name = options.name+"_sig%.1f"%sigma+("_fs" if full_supervision else "")
        #for i,(train_i,val_i) in enumerate(kf.split(train)):
        for i in range(n_folds):
                np.random.shuffle(train)
                train_i = np.random.choice(len(train),int(0.8*len(train)), replace=False)
                val_i = np.delete(np.arange(len(train)),train_i)
                train_kf, val_kf = train[train_i], train[val_i]
                print("10th train sample",train_kf[10])
                print("10th val sample",val_kf[10])
                if not np.any(val_kf==0): print("THERE ARE NO BKG EVENTS IN THE VAL SET")
                print(">> Training BDT with %ith fold as validation"%i)
                #bdt = HGBClassifier(max_iters=None, early_stopping=True, verbose=True)
                #bdt.fit(train_kf[:,:n_features],train_kf[:,n_features], val_kf[:,:n_features], val_kf[:,n_features])
                bdt = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, learning_rate = 0.2, validation_fraction=0.2, warm_start=True)
                #bdt = RandomForestClassifier(n_estimators = 200, warm_start = True)
                bdt.fit(train[:,:n_features],train[:,n_features])
                pred_list = bdt.predict_proba(test[:,:n_features])[:,1]
                #print("Validation score per iteration: ",bdt.validation_score_)
                pred_list_all.append(pred_list)
        pred_list_all = np.array(pred_list_all)
        print("All results shape: ",pred_list_all.shape)
        pred_list = np.mean(pred_list_all, axis=0)
        print("After averaging all the results: ",pred_list)
        true_list = test[:,-1]
        print("After averaging results of kfold, predicted list shape = ", pred_list.shape)
        np.savetxt("losses/fpr_tpr_bdt_%s%s_N%i.txt"%(name,("_fs" if full_supervision else ""),n_folds),np.vstack((true_list,pred_list)))
        
        
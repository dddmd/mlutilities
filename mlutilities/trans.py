def pca_trans(train, test, cols, n_comp, prefix='pca_', seed=0, svd_solver='randomized', pca_obj=None):
    
    #svd_solver:'randomized' or'full'
    if pca_obj == None: #INFERENCE:
        pca=PCA(n_components=n_comp, random_state=seed,svd_solver=svd_solver)
    else:
        pca=pca_obj

    data = pd.concat([pd.DataFrame(train[cols]), pd.DataFrame(test[cols])])
    pca.fit(data[cols])
    data2  = pca.transform(data[cols])
    train2 = data2[:train.shape[0]]; test2 = data2[-test.shape[0]:]
    train2 = pd.DataFrame(train2, columns=[f'{prefix}-{i}' for i in range(n_comp)])
    test2  = pd.DataFrame(test2, columns=[f'{prefix}-{i}' for i in range(n_comp)])
    print(f'explained_variance_ratio_ sum:{np.sum(pca.explained_variance_ratio_):.7f},max:{pca.explained_variance_ratio_[0]:.7f},min:{pca.explained_variance_ratio_[-1]:.7f}')
    return train2, test2, pca
	
	#train  = pd.concat((train, train2), axis=1)
    #test   = pd.concat((test, test2), axis=1)
    #return train, test, pca
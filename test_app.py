import unittest
import pandas as pd
import os

class TestUnitaire(unittest.TestCase):

    ###################################
    ## EXISTANCE DES FICHIERS        ##
    ###################################
    def test_exists_dict_columns(self):
        print('Test: Asserting existance of columns_dict.pkl')
        exist = os.path.exists("./Data/Processed_data/columns_dict.pkl")
        self.assertEqual(exist, True)
        
    def test_exists_test_df(self):
        print('Test: Asserting existance of test_df.pkl')
        exist = os.path.exists("./Data/Processed_data/test_df.pkl")
        self.assertEqual(exist, True)

    def test_exists_train_df(self):
        print('Test: Asserting existance of application_test.pkl')
        exist = os.path.exists('./Data/Processed_data/application_test.pkl')
        self.assertEqual(exist, True)

    def test_exists_app_train(self):
        print('Test: Asserting existance of train_df.pkl')
        exist = os.path.exists('./Data/Processed_data/train_df.pkl')
        self.assertEqual(exist, True)

    def test_exists_shap_values(self):
        print('Test: Asserting existance of 230616_shap_values.pkl')
        exist = os.path.exists('./Data/Processed_data/230616_shap_values.pickle')
        self.assertEqual(exist, True)


    
    
    ###################################
    ## TEST NOMBRE DE COLONNEs        ##
    ###################################
    
    def test_numcol_column_test_set(self):
        print('Test: Asserting nb of columns in  of test_df.pkl')
        test_df = pd.read_pickle('./Data/Processed_data/test_df.pkl')
        result = test_df.shape[1]
        self.assertEqual(result, 546)
        
    def test_numcol_app_test_df(self):
        print('Test: Asserting nb of columns in  of application_test.pkl')
        app_test_df = pd.read_pickle('./Data/Processed_data/application_test.pkl')
        result = app_test_df.shape[1]
        self.assertEqual(result, 121)
        
    def test_numcol_train_df(self):
        print('Test: Asserting nb of columns in  of train_df.pkl')
        train_df = pd.read_pickle('./Data/Processed_data/train_df.pkl')
        result = train_df.shape[1]
        self.assertEqual(result, 546)
        
    def test_numcol_shap_values(self):
        print('Test: Asserting nb of columns in  of 230616_shap_values.pkl')
        shape_values = pd.read_pickle('./Data/Processed_data/230616_shap_values.pickle')
        result = shape_values.shape[1]
        self.assertEqual(result, 159)

    

        
    ################################
    ## TEST  NOMS DES COLONNES    ##
    ################################
    
    
    def test_namecol_column_test_set(self):
        print('Test: Asserting columns name in  of test_df.pkl')
        test_df = pd.read_pickle('./Data/Processed_data/test_df.pkl')
        ref = pd.read_pickle("./Data/Processed_data/columns_dict.pkl")['columns_test_df']
        result = test_df.columns.tolist()
        assert all([a == b for a, b in zip(ref, result)])

    def test_namecol_column_application_test(self):
        print('Test: Asserting columns name in  of application_test.pkl')
        app_test_df = pd.read_pickle('./Data/Processed_data/application_test.pkl')
        ref = pd.read_pickle("./Data/Processed_data/columns_dict.pkl")['columns_application_test']
        result = app_test_df.columns.tolist()
        assert all([a == b for a, b in zip(ref, result)])

    def test_namecol_column_train_df(self):
        print('Test: Asserting columns name in  of train_df.pkl')
        train_df = pd.read_pickle('./Data/Processed_data/train_df.pkl'))
        ref = pd.read_pickle("./Data/Processed_data/columns_dict.pkl")['columns_train_df']
        result = train_df.columns.tolist()
        assert all([a == b for a, b in zip(ref, result)])

        
if __name__ == '__main__':
    unittest.main()
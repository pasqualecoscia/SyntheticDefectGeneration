import os
import shutil
import argparse
import random

def copy_files(src_folder, dest_folder, mask_folder=None, mask_dest_folder=None, file_list=None): 
    '''
    Copy all files from src_folder to dest_folder.
    If add_folder is provided, its content is copied into dest_folder
    '''
    src_files = os.listdir(src_folder)
    for file_name in src_files:
        full_file_name = os.path.join(src_folder, file_name)
        if os.path.isfile(full_file_name) and (file_list is None or file_name in file_list):
            # Rename if file exists
            num_file = int(os.path.splitext(os.path.basename(full_file_name))[0])
            ext_file = os.path.splitext(os.path.basename(full_file_name))[-1]
            dest_folder_file_renamed = os.path.join(dest_folder, f"{num_file:03}" + ext_file)
            while os.path.exists(dest_folder_file_renamed):
                num_file += 1
                dest_folder_file_renamed = os.path.join(dest_folder, f"{num_file:03}" + ext_file)
            shutil.copy(full_file_name, dest_folder_file_renamed)
            if mask_folder is not None:
                end_name = '_mask' + ext_file
                MASK_FILE_PATH = os.path.join(mask_folder, full_file_name.split('/')[-1].split('.')[0] + end_name)
                MASK_DEST_FOLDER = os.path.join(mask_dest_folder, MASK_FILE_PATH.split('/')[-1])
                shutil.copy(MASK_FILE_PATH, MASK_DEST_FOLDER)

class MVTECDataset:
    # Class to create the MVTEC dataset using the same format as cycle-gan dataset

    def __init__(self, PATH):
        # Extract products and defects
        
        # Dataset Path
        self.path = PATH

        # Dictionary containing for each product its corresponding defects
        self.products_defects_dict = self.extract_products_defects(self.path)

        self.selected_products = []
        self.selected_defects = []
    
    def extract_products_defects(self, mvtec_path):
        '''
        Returns a dictionary where keys are the products found in mvtec_path and 
        values are the corresponding defects
        '''
        prods_defs_dict = dict()
        
        for product in os.listdir(mvtec_path):
            PRODUCT_PATH = os.path.join(mvtec_path, product)
            if os.path.isdir(PRODUCT_PATH):
                # Defects of product "product"
                defects_list = [defect for defect in os.listdir(os.path.join(PRODUCT_PATH, "test")) if defect != "good"]                
                # Add product and defect to dict                
                prods_defs_dict[product] = defects_list
        
        return prods_defs_dict

    def check_inputs(self, product, defect):
        '''
        Check if input product name or defect name is correct
        Output:
            True if correct
            False if either product or defect name is not correct

        This function also creates selected_products and selected_defects lists.
        '''
        # Boolean variables that check if product and defect names are correct
        _product = False
        _defect = False

        prod_def_dict = self.products_defects_dict
        
        #TODO: implement specific defect for each product
        if product in prod_def_dict.keys():
            _product = True
            self.selected_products.append(product)
            if (defect in prod_def_dict[product]):
                _defect = True
                self.selected_defects.append(defect)
            elif defect == 'one':
                _defect = True
                self.selected_defects.append(random.choice(prod_def_dict[product]))
            elif defect == 'all':
                _defect = True
                self.selected_defects = prod_def_dict[product]
        # elif product == 'all':
        #     _product = True
        #     self.selected_products = prod_def_dict.keys()
        #     if defect == 'one':
        #         _defect = True
        #         for p in self.selected_products:
        #             self.selected_defects.append(random.choice(prod_def_dict[p]))                
        #     elif defect == 'all': 
        #         _defect = True
        
        return (_product and _defect)


    def save_dataset(self, product, defect):
        '''
        Save dataset in the following format:
        
        DATA
        |------PRODUCT_NAME
               |-------------TRAIN
               |               |-----A
               |               |-----B
               |               |-----mask

               |-------------TEST  
               |               |-----A
               |               |-----B
               |               |-----mask

        '''

        # Check if product name and defect name are correct
        # Note: 'all' means all products (or all defects)
        if not self.check_inputs(product, defect):
            raise ValueError('Product or defect name not recognized. Please, check input values.')

        print(f"Selected product: {self.selected_products}")
        print(f"Selected defect: {self.selected_defects}")

        # Define dataset path './data/mvtec_dataset'
        DATASET_PATH = os.path.join(os.path.dirname(self.path), 'mvtec_dataset')

        # Create train and test folders
        TRAIN_A_PATH = os.path.join(DATASET_PATH, "train/A")
        TRAIN_B_PATH = os.path.join(DATASET_PATH, "train/B")
        TRAIN_MASK_PATH = os.path.join(DATASET_PATH, "train/mask")
        TEST_A_PATH = os.path.join(DATASET_PATH, "test/A")
        TEST_B_PATH = os.path.join(DATASET_PATH, "test/B")
        TEST_MASK_PATH = os.path.join(DATASET_PATH, "test/mask")        

        try:
           os.makedirs(DATASET_PATH)
           os.makedirs(TRAIN_A_PATH)
           os.makedirs(TRAIN_B_PATH)
           os.makedirs(TRAIN_MASK_PATH)
           os.makedirs(TEST_A_PATH)
           os.makedirs(TEST_B_PATH)
           os.makedirs(TEST_MASK_PATH)           
        except OSError:
            pass
        
        for product in self.selected_products:
            # DOMAIN A IMAGES
            TRAIN_A_IMAGES_PATH = os.path.join(self.path, product, "train/good")
            TEST_A_IMAGES_PATH = os.path.join(self.path, product, "test/good")
            copy_files(src_folder=TRAIN_A_IMAGES_PATH, dest_folder=TRAIN_A_PATH)
            copy_files(src_folder=TEST_A_IMAGES_PATH, dest_folder=TEST_A_PATH)
            
            for defect in self.selected_defects:                               

                # DOMAIN B IMAGES
                DOMAIN_B_PATH = os.path.join(self.path, product, "test", defect)
                DOMAIN_B_PATH_MASK = os.path.join(self.path, product, "ground_truth", defect)
                images_b_list = os.listdir(DOMAIN_B_PATH)
                images_b_list.sort()
                # Select 80/20 for train/test
                
                train_b_list = images_b_list[:round(len(images_b_list) * 0.8)]
                test_b_list = images_b_list[round(len(images_b_list) * 0.8):]

                # Copy files
                copy_files(src_folder=DOMAIN_B_PATH, dest_folder=TRAIN_B_PATH, mask_folder=DOMAIN_B_PATH_MASK, mask_dest_folder=TRAIN_MASK_PATH, file_list=train_b_list)
                copy_files(src_folder=DOMAIN_B_PATH, dest_folder=TEST_B_PATH, mask_folder=DOMAIN_B_PATH_MASK, mask_dest_folder=TEST_MASK_PATH, file_list=test_b_list)

if __name__ == "__main__":
    '''
    Create MVTEC dataset using the same format as cycle-gan code.
    Four different cases are considered (P stands for product while D for defect):

    1.  1 P - 1 D -> one product and one defect
    1.  1 P - N D -> one product and N defects (N -> all)
    --------------------------------------------- IMPLEMENTED THIS FOR NOW

    1.  N P - 1 D -> N products and one defect per product (the defect is randomly chosen) TODO: select specific defect
    1.  N P - N D -> N products and N defects

    '''
    parser = argparse.ArgumentParser(
    description="MVTEC dataset creator. You can optionally specify product and defect.")
    parser.add_argument("--product", type=str, default="transistor", help="Product. (default:`transistor`). If 'all', all products are selected.")
 
    parser.add_argument("--defect", type=str, default="all", help="Defect. (default:`all`). Values [one|all|defect_name]. If 'all', all defects are used and combined. If 'one' a defect is randomly selected. If 'defect_name', the selected defect is used.")
    args = parser.parse_args()

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    MVTEC_PATH = os.path.join(CURRENT_PATH, 'mvtec')

    mvtect_dataset = MVTECDataset(MVTEC_PATH)
    mvtect_dataset.save_dataset(args.product, args.defect)

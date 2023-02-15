from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import os
from typing import Any,Dict,List,Optional


doctr_model = ocr_predictor(pretrained=True)



class OCR:
    def __init__(self,ocr:str):
        #self.file = file
        self.text:Dict = {}
        self.data: Dict  = {}
        self.structered_text : Dict = {}
        self.ocr = ocr
        self.dimensions: Dict = {}
        self.data_line: Dict = {}
        self.text_line : Dict = {}

    def get_data(self,file:Any):
        filetype = self.__check_file_type(file)
        if filetype == "pdf":
            docs = DocumentFile.from_pdf(file)
        elif filetype == "image":
            docs = DocumentFile.from_images(file)


        if self.ocr== 'doctr':
            model = doctr_model
        elif self.ocr== "tesseract":
            model = pytesseract
        
        for i,doc in enumerate(docs):
            result = self.extract_text(doc,self.ocr)
            standard_result = self.standardized_op(result,self.ocr)
            self.data[i] = standard_result
            temp_dict = {}
            temp_dict['width'] = doc.shape[1]
            temp_dict['height']=doc.shape[0]
            self.dimensions[i]=temp_dict
            raw_text = " ".join(standard_result['text'])
            self.text[i] = raw_text
            ocr_line_df = self.ocr_dump_to_line_df(standard_result)
            self.data_line[i] = ocr_line_df
            raw_text_line = "\n".join(ocr_line_df['text'])
            self.text_line[i] = raw_text_line
        
        return self.data,self.dimensions
 
    def get_text(self,file:Any) -> str:
        data,_ = self.get_data(file)
        # for k,v in data.items():
        #     raw_text = " ".join(v['text'])
        #     self.text[k] = raw_text
        # return self.text,self.dimensions
        return self.text
    
    def get_line_data(self,file:Any):
        data,_ = self.get_data(file)
        return self.data_line

    def get_line_text(self,file:Any):
        data,_ = self.get_data(file)
        return self.text_line
    
    def ocr_dump_to_line_df(self,ocr_df):
        line_list = []
        for i,(idx,row) in enumerate(ocr_df.iterrows()):
            if i == 0:
                prev_row = row
                min_left = row['left']
                min_top = row['top']
                max_right = row['left'] + row['width']
                max_down = row['top'] + row['height']
                row_text = row['text']
                # temp_row.append[]
            elif prev_row['line_num'] != row['line_num']:
                line_list.append([row['block_num'],prev_row['line_num'],min_left,min_top,max_right,max_down,row_text])
                prev_row = row
                min_left = row['left']
                min_top = row['top']
                max_right = row['left'] + row['width']
                max_down = row['top'] + row['height']
                row_text = row['text']
            else:
                min_left = min(min_left,row['left'])
                min_top = min(min_top,row['top'])
                max_right = max(max_right,(row['left'] + row['width']))
                max_down = max(max_down,(row['top'] + row['height']))
                row_text =row_text+" "+row['text']
            if i == len(ocr_df) - 1:
                line_list.append([row['block_num'],row['line_num'],min_left,min_top,max_right,max_down,row_text])
        ocr_line_df = pd.DataFrame(line_list,columns=['block_num','line_num','left','top','right','down','text'])
        return ocr_line_df




    def get_processed_text(self) -> Dict:
        return self.text

    def standardized_doctr_output(self,result):
        doctr_lst=[]
        pge_cnt = 1
        for page in result.pages:
                blck_cnt = 1
                for block in page.blocks:
                    lin_cnt=1
                    for line in block.lines:
                        word_cnt = 1
                        for word in line.words:
                            word.geometry
                            temp_arr = []
                            temp_arr.append(blck_cnt)
                            temp_arr.append(lin_cnt)
                            temp_arr.append(word_cnt)
                            xmin_raw = word.geometry[0][0]
                            ymin_raw = word.geometry[0][1]
                            xmax_raw = word.geometry[1][0]
                            ymax_raw = word.geometry[1][1]
                            xmin_geo=int(xmin_raw*page.dimensions[1])
                            ymin_geo=int(ymin_raw*page.dimensions[0])
                            xmax_geo=int(xmax_raw*page.dimensions[1])
                            ymax_geo=int(ymax_raw*page.dimensions[0])
                            width = abs(xmax_geo-xmin_geo)
                            height = abs(ymax_geo-ymin_geo)
                            temp_arr.append(xmin_geo)
                            temp_arr.append(ymin_geo)
                            temp_arr.append(xmax_geo)
                            temp_arr.append(ymax_geo)
                            temp_arr.append(width)
                            temp_arr.append(height)
                            temp_arr.append(word.confidence)
                            temp_arr.append(word.value)
                            doctr_lst.append(temp_arr)
                            word_cnt+=1
                        lin_cnt+=1
                    blck_cnt+=1
        cols = ["block_num","line_num","word_num","left","top","right","down","width","height","conf","text"]
        doctr_df = pd.DataFrame(doctr_lst,columns=cols)
        return doctr_df

    def standardize_tesseract_output(self,df):
        df = df [df['conf']>0]
        df.loc[:,'right'] = df.loc[:,'left'] + df.loc[:,'width']
        df.loc[:,'down'] = df.loc[:,'top'] + df.loc[:,'height']
        tesseract_df = df[["block_num","line_num","word_num","left","top","right","down","width","height","conf","text"]]
        return tesseract_df

    def standardized_op(self,result,ocr_method):
        if ocr_method == 'doctr':
            result = self.standardized_doctr_output(result)
        elif ocr_method == "tesseract":
            result= self.standardize_tesseract_output(result)
        result = self.__sort_result(result)
        return result
    
    def extract_text(self,doc,ocr_method):
        if ocr_method == 'doctr':
            result = doctr_model([doc])
        elif ocr_method == "tesseract":
            result = pd.DataFrame(pytesseract.image_to_data(doc,output_type=pytesseract.Output.DICT))
        return result
    
    def __check_file_type(self,file:Any):
        filetype:str = ''
        if os.path.splitext(file)[-1] == ".pdf":
            filetype = "pdf"
        else:
            filetype = "image"
        return filetype

    def __sort_result(self,result):
        rs_sorted = result.sort_values(by = ['top'])
        rs_sorted.reset_index(drop=True,inplace=True)
        final_df = pd.DataFrame(columns=['block_num', 'line_num', 'word_num', 'left', 'top', 'right', 'down','width', 'height', 'conf', 'text'])
        block_df = pd.DataFrame(columns=['block_num', 'line_num', 'word_num', 'left', 'top', 'right', 'down','width', 'height', 'conf', 'text'])
        prev_top = ''
        row_num = 0
        for index, row in rs_sorted.iterrows():
            if index==0:
                row['line_num'] = row_num
                block_df = block_df.append(row,ignore_index=True)
                prev_top = row['top']
            if index>0:
                if abs(row['top'] - prev_top)>15:
                    block_df_sorted = block_df.sort_values(by=['left'])
                    final_df =final_df.append(block_df_sorted,ignore_index=True)
                    row_num+=1
                    block_df = pd.DataFrame()
                    row['line_num'] = row_num
                    block_df = block_df.append(row,ignore_index=True)
                    prev_top = row['top']
                else:
                    row['line_num'] = row_num
                    block_df = block_df.append(row)
                    prev_top = row['top']
        block_df_sorted = block_df.sort_values(by=['left'])
        final_df =final_df.append(block_df_sorted,ignore_index=True)
        return final_df



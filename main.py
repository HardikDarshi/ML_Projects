from flask import Flask, render_template, request, make_response, send_file
import pandas as pd
import pickle
import os

import Preprocessing

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            f = request.files['file1']
            df = pd.read_excel(f)
            df.columns = df.columns.str.replace(' ', '')
            columns = ['id', 'MRBTS', "redirFreqCdma", "Item-redirGeranArfcnStructL-redirGeranArfcnPrio",
                       'redirGeranArfcnStructL', 'redirGeranBandIndicator',
                       'Item-redirGeranArfcnStructL-redirGeranArfcnValue']
            preprocessor = Preprocessing.Preprocessor()

            df = preprocessor.handle_missing_values(df)
            df = preprocessor.remove_columns(df,columns)
            filename1 = 'DTmodel_Pred_REDRTV1.sav'
            loaded_model = pickle.load(open(filename1, 'rb'))
            prediction = loaded_model.predict(df)
            result = pd.DataFrame(prediction)
            result.columns = ['Pred']
            result = preprocessor.int_to_categorical(result)
            final_sheet = pd.merge(df, result, left_index = True, right_index = True)
            print('final sheet is prepared')
            resp = make_response(final_sheet.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp
            #final_sheet1= final_sheet.to_excel(r'C:\Users\CNU23895TK\PycharmProjects\REDRT2\uploads\outputfile.xlsx', index=True, header= True)
            #final_sheet.to_excel(r'C:\Users\CNU23895TK\PycharmProjects\REDRT2\uploads\outputfile.xlsx', index = True, header = True)
            #final_sheet1.save(os.path.join("uploads", final_sheet1.filename))

            #return 'Output file is saved to uploads folder'
        except Exception as e:
            print('The Exception  message is:', e)
            return 'Something is Wrong'
    else:
        return render_template('index.html')

@app.route('/download')
def downlaod_reffile():
    p= 'reference_file.xlsx'
    return send_file(p, as_attachment= True)

if __name__ == '__main__':
    #app.run(debug = True)
    #app.run(host='127.0.0.1', port=5001, debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)

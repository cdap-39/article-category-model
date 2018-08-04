# from flask import Flask,jsonify,request,make_response,url_for,redirect
# import requests, json
#
# app = Flask(__name__)
#
# url = 'https://hooks.zapier.com/hooks/catch/xxxxx/yyyyy/'
#
# @app.route('/', methods=['GET','POST'])
# def create_row_in_gs():
#     if request.method == 'GET':
#         return make_response('failure')
#     if request.method == 'POST':
#         print("POST IN")
#         json_data = request.get_json(force=True)
#         #t_id = request.json()#['data']
#         print(json_data)
#         # t_name = request.json['name']
#         # created_on = request.json['created_on']
#         # modified_on = request.json['modified_on']
#         # desc = request.json['desc']
#         #
#         # create_row_data = {'id': str(t_id),'name':str(t_name),'created-on':str(created_on),'modified-on':str(modified_on),'desc':str(desc)}
#         row_data={"HI":"SANDY"}
#         response = requests.post(
#             url, data=json.dumps(row_data),
#             headers={'Content-Type': 'application/json'}
#         )
#         return response.content
#
# if __name__ == '__main__':
#     #app.run(host='localhost',debug=False, use_reloader=True)
#
#     app.run(host='0.0.0.0', port=8000, debug=True)
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

#parser = reqparse.RequestParser()
#parser.add_argument('username', type=unicode, location='json')
#parser.add_argument('password', type=unicode, location='json')

class HelloWorld(Resource):
    def post(self):
        print("HI")
        json_data = request.get_json(force=True)
        print(json_data['data'])
        if not json_data:
               return {'message': 'No input data provided'}, 400

        # un = json_data['username']
        # pw = json_data['password']
        #args = parser.parse_args()
        #un = str(args['username'])
        #pw = str(args['password'])
        return jsonify(json_data)

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000 ,debug=True)




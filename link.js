
var mysql      = require('mysql');
var connection = mysql.createConnection({
  host     : 'localhost',
  user     : 'root',
  password : '941012',
  database : 'test0'
});
 
connection.connect();

connection.query('SELECT T1_P2SEL FROM test0.test where id = 1;', function (error, result) {
  if (error) throw error;
  console.log(result[0].T1_P2SEL);
  callback(result[0].T1_P2SEL);
})

var _getUser = function(name, callback) {


    connection.query(sql, function(err, results) {
        if(!err) {
            var res = hasUser(results)
            callback(res);

        }else {
            callback(error());
        }
    });

}

var getUser = function(name, callback){
    return _getUser(name, callback);
}


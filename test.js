 console.log("Hello World"); 



connection.query('SELECT T1_P2SEL FROM test0.test where id = 1;', function (error, result) {
  if (error) throw error;
  console.log(result[0].T1_P2SEL);
  callback(result);
})
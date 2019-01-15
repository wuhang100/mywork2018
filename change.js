var mycontext=""

function displayDate(){
	var a = document.getElementById("demo2").innerHTML;
	if (a < 2){
		a = 2;
		document.getElementById("demo2").innerHTML=a;
		document.getElementById("demo").innerHTML='<img src="/picture/timg.jpg" alt="a dog" height="200" width="280"/>';
	}
	else {
		a = 1;
		document.getElementById("demo2").innerHTML=a;
		document.getElementById("demo").innerHTML='<img src="/picture/dog.jpg" alt="a dog" height="200" width="280"/>';
	}
}
	
function choosedog(){
    var select = document.getElementById("s1");
    var value = select.value;
    var options = select.options;
    var index = select.selectedIndex;
    var selectedText = options[index].text;
	mycontext = mycontext + "," + selectedText;
	document.getElementById("demo3").innerHTML = mycontext;
}	

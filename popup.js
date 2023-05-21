document.addEventListener('DOMContentLoaded', function () {
  var form = document.getElementById('newsForm');
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var newsTitle = document.getElementById('newsInput').value;
    console.log(newsTitle);
    if (newsTitle.trim() !== '') {
      //Disable button
      document.getElementById('submitButton').disabled = true;
      //load spinner
      document.getElementById('loading').style.display = "block";
      //Hide results div (if visible)
      document.getElementById('results').style.display = "none";
      sendRequest(newsTitle);
    }
  });
});

function sendRequest(newsTitle) {
  var url = 'http://127.0.0.1:7000/predict'; // Replace with your backend API URL
  var xhr = new XMLHttpRequest();
  xhr.open('POST', url, true);
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.onreadystatechange = function () {
    if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
      console.log(response)
      var response = JSON.parse(xhr.responseText);
      displayResults(response);
    } else {
      handleError();
    }
  };
  var data = 'title=' + encodeURIComponent(newsTitle);
  xhr.send(data);
}

function displayResults(scores) {
  //enable button
  document.getElementById('submitButton').disabled = false;
  //hide spinner
  document.getElementById('loading').style.display = "none";

  console.log(scores)

  // parseFloat("123.456").toFixed(2);

  document.getElementById('support-text').textContent = `Support: ${(scores.entailment * 100).toFixed(2)}%`;
  document.getElementById('neutral-text').textContent = `Neutral: ${(scores.neutral * 100).toFixed(2)}%`;
  document.getElementById('contradiction-text').textContent = `Contradiction: ${(scores.contradiction * 100).toFixed(2)}%`;

  const sAngle = scores.entailment * 360;
  const nAngle = scores.neutral * 360;

  document.getElementById('pChart').style.backgroundImage = `conic-gradient(
    #5cb85c 0deg,
    #5cb85c ${sAngle}deg,
    #f0ad4e ${sAngle}deg,
    #f0ad4e ${sAngle + nAngle}deg,
    #d9534f ${sAngle + nAngle}deg)`;

  //Show Results
  document.getElementById('results').style.display = "block";
}

//ERROR HANDLING???

function handleError() {
  //enable button
  document.getElementById('submitButton').disabled = false;
  //hide spinner
  document.getElementById('loading').style.display = "none";
  //Hide results div (if visible)
  document.getElementById('results').style.display = "none";
  window.alert("Error Fetching Results");
}
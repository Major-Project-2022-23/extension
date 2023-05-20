document.addEventListener('DOMContentLoaded', function() {
    var form = document.getElementById('newsForm');
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      var newsTitle = document.getElementById('newsTitle').value;
      if (newsTitle.trim() !== '') {
        sendRequest(newsTitle);
      }
    });
  });
  
  function sendRequest(newsTitle) {
    var url = 'http://127.0.0.1:7000/predict'; // Replace with your backend API URL
    var xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log(response)
        var response = JSON.parse(xhr.responseText);
        displayResults(response);
      }
    };
    var data = 'title=' + encodeURIComponent(newsTitle);
    xhr.send(data);
  }
  
  function displayResults(scores) {
    console.log(scores)
    console.log(scores.contradiction)
    var resultContainer = document.getElementById('resultContainer');
    var ContradictionScore = document.getElementById('ContradictionScore');
    var EntailmentScore = document.getElementById('EntailmentScore');
    var NeutralScore = document.getElementById('NeutralScore');
    
    ContradictionScore.textContent = 'Fake Score: ' + scores.contradiction;
    EntailmentScore.textContent = 'Neutral Score: ' + scores.entailment;
    NeutralScore.textContent = 'True Score: ' + scores.neutral;
    
    resultContainer.style.display = 'block';
  }
  
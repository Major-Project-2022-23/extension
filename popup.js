$(document).ready(function() {
  $('#submit-btn').click(function(event) {
    event.preventDefault(); // Prevent the default form submission

    var title = $('#title-input').val(); // Get the value from the input field

    $.ajax({
      url: 'http://127.0.0.1:7000/predict',
      type: 'POST',
      data: { title: title },
      success: function(response) {
        // Display the prediction scores in the #prediction-scores div
        $('#prediction-scores').html('Contradiction: ' + response.contradiction + '<br>' +
                                      'Entailment: ' + response.entailment + '<br>' +
                                      'Neutral: ' + response.neutral);
      },
      error: function(error) {
        console.log(error);
      }
    });
  });
});

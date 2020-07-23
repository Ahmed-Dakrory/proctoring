function gcd(a,b)

{
        if (a<0.00000001)

        {

                return b;

        }

        if (a<b)

        {

                return gcd(b-Math.floor(b/a)*a,a);

        }

        else if (a==b)

        {

                return a;

        }

        else

        {

                return gcd(b,a);

        }

}

 

var x_init=performance.now()/1000;

var g=performance.now()/1000-x_init;

for (var i=0;i<10;i++)

{

        g=gcd(g,performance.now()/1000-x_init);

}

//alert("Performance Counter Frequency: "+Math.round(1/g)+" Hz");


audio = false;
mic = false;
cam=false;

function gotDevices(deviceInfos) {
  // Handles being called several times to update labels. Preserve values.


  for (let i = 0; i !== deviceInfos.length; ++i) {
    const deviceInfo = deviceInfos[i];
    if(deviceInfo.kind=='audioinput'){
      mic = true;
    }else if(deviceInfo.kind=='audiooutput'){
      audio = true;
    }else if(deviceInfo.kind=='videoinput'){
      cam = true;
    }
    
  }
  
}


function handleError(error) {
  console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
}
navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);


$("#devices_check").click(function(){
  setTimeout(function(){
    //do what you need here
    if(audio){
      $( "#speaker_i" ).last().removeClass( "fa-refresh" );
      $( "#speaker_i" ).last().addClass( "fa-check active" );
      setTimeout(function(){
        if(mic){
          $( "#mic_i" ).last().removeClass( "fa-refresh" );
          $( "#mic_i" ).last().addClass( "fa-check active" );
          setTimeout(function(){
            if(cam){
              $( "#camera_i" ).last().removeClass( "fa-refresh" );
              $( "#camera_i" ).last().addClass( "fa-check active" );
            }
        }, 500);
          
          
        }
    }, 500);
      
    }
   
}, 500);
  
});

var width = 320;    // We will scale the photo width to this
var height = 0;     // This will be computed based on the input stream

var streaming = false;

var video = null;
var canvas = null;
var photo = null;
var startbutton = null;

video = document.getElementById('video');
canvas = document.getElementById('canvas');
//photo = document.getElementById('photo');
startbutton = document.getElementById('startbutton');


    video.addEventListener('canplay', function(ev){
      if (!streaming) {
        height = video.videoHeight / (video.videoWidth/width);
      
        video.setAttribute('width', width);
        video.setAttribute('height', height);
        streaming = true;
      }
    }, false);

    
    function clearphoto() {
      var context = canvas.getContext('2d');
      context.fillStyle = "#AAA";
      context.fillRect(0, 0, canvas.width, canvas.height);
  
      var data = canvas.toDataURL('image/png');
      //photo.setAttribute('src', data);
    }


    var valuesTobeSend;
    function takepicture() {
      var context = canvas.getContext('2d');
      if (width && height) {
        canvas.width = width;
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);
        
        var image = canvas.toDataURL('image/png');
        valuesTobeSend ={"image":image}
        
        
        //photo.setAttribute('src', data);
      } else {
        clearphoto();
      }
    }

    function runIdentification(){
      $.ajax({
        url: "/imgPosting" ,
        data: valuesTobeSend,
        type: 'POST',
        success: function (dataR) {
          
          console.log(dataR);
          if(dataR.verfied=='True'){
            $( "#verficationI" ).last().removeClass( "fa-times" );
            $( "#verficationI" ).last().addClass( "fa-check" );

            $( "#verficationIDiv" ).last().removeClass( "falseMan" );
            $( "#verficationIDiv" ).last().addClass( "trueMan" );
          }else{
            $( "#verficationI" ).last().removeClass( "fa-check" );
            $( "#verficationI" ).last().addClass( "fa-times" );

            
            $( "#verficationIDiv" ).last().removeClass( "trueMan" );
            $( "#verficationIDiv" ).last().addClass( "falseMan" );
          }
        },
        error: function (xhr, status, error) {
            console.log('Error: ' + error.message);
        },
        timeout: 3000 // sets timeout to 3 seconds
    });
    }


    var refreshIntervalId ;

    $("#endbutton").click(function(){
      // A video's MediaStream object is available through its srcObject attribute
const mediaStream = video.srcObject;

// Through the MediaStream, you can get the MediaStreamTracks with getTracks():
const tracks = mediaStream.getTracks();

// Tracks are returned as an array, so if you know you only have one, you can stop it with: 
tracks[0].stop();

// Or stop all like so:
tracks.forEach(track => track.stop())
clearInterval(refreshIntervalId);
clearInterval(runIdentificationIntervalId);
});


$("#startbutton").click(function(){
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then(function(stream) {
      video.srcObject = stream;
      video.play();
  })
  .catch(function(err) {
      console.log("An error occurred: " + err);
  });

 refreshIntervalId = setInterval(function(){
  takepicture();

}, 150);


runIdentificationIntervalId = setInterval(function(){
  runIdentification();

}, 3000);
});
    
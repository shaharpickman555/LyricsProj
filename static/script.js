document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('fileInput').addEventListener('change', function(event) {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = URL.createObjectURL(file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Transcription Data:', data);  // Debug: Log the transcription data

            const transcriptionDiv = document.getElementById('transcription');
            transcriptionDiv.innerHTML = '';

            const isEnglish = data.language === 'en';  // Check if the language is English
            const isHebrew = data.language === 'he';  // Check if the language is Hebrew

            // Apply the RTL class for Hebrew
            if (isHebrew) {
                transcriptionDiv.classList.add('rtl');
            } else {
                transcriptionDiv.classList.remove('rtl');
            }

            if (data.segments) {
                const words = [];

                data.segments.forEach(segment => {
                    if (segment.words) {
                        segment.words.forEach(word => {
                            words.push({
                                text: word.word,
                                start: word.start,
                                end: word.end
                            });
                        });
                    } else {
                        console.warn('No words found in segment:', segment);
                    }
                });

                console.log('Words:', words);  // Debug: Log the words array

                words.forEach(word => {
                    if (word && word.text) {
                        const span = document.createElement('span');
                        span.textContent = word.text + ' ';
                        span.dataset.start = word.start;
                        span.dataset.end = word.end;

                        // Add newline before capital letters if the language is English
                        if (isEnglish && word.text[0] === word.text[0].toUpperCase() && transcriptionDiv.innerHTML !== '') {
                            transcriptionDiv.appendChild(document.createElement('br'));
                        }

                        transcriptionDiv.appendChild(span);
                    } else {
                        console.warn('Invalid word:', word);
                    }
                });

                let lastHighlightedSpan = null;

                audioPlayer.addEventListener('timeupdate', function() {
                    const currentTime = audioPlayer.currentTime;
                    console.log('Current Time:', currentTime);  // Debug: Log the current playback time

                    let highlightFound = false;
                    document.querySelectorAll('#transcription span').forEach(span => {
                        const start = parseFloat(span.dataset.start);
                        const end = parseFloat(span.dataset.end);

                        if (currentTime >= start && currentTime <= end) {
                            span.classList.add('highlight');
                            lastHighlightedSpan = span;
                            highlightFound = true;
                        } else {
                            span.classList.remove('highlight');
                        }
                    });

                    // Keep highlighting the last detected word during instrumental sections
                    if (!highlightFound && lastHighlightedSpan) {
                        lastHighlightedSpan.classList.add('highlight');
                    }
                });
            } else {
                console.error('No segments found in the response:', data);
            }
        })
        .catch(error => {
            console.error('Error:', error);  // Debug: Log any errors
        });
    });
});






// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('fileInput').addEventListener('change', function(event) {
//         const file = event.target.files[0];
//         const formData = new FormData();
//         formData.append('file', file);
//
//         const audioPlayer = document.getElementById('audioPlayer');
//         audioPlayer.src = URL.createObjectURL(file);
//
//         fetch('/upload', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log('Transcription Data:', data);  // Debug: Log the transcription data
//
//             const transcriptionDiv = document.getElementById('transcription');
//             transcriptionDiv.innerHTML = '';
//
//             if (data.segments) {
//                 const segments = data.segments;
//                 const words = [];
//
//                 segments.forEach(segment => {
//                     if (segment.text && typeof segment.start === 'number' && typeof segment.end === 'number') {
//                         const segmentWords = segment.text.split(' ').map((word, index, arr) => {
//                             const wordDuration = (segment.end - segment.start) / arr.length;
//                             return {
//                                 text: word,
//                                 start: segment.start + (index * wordDuration),
//                                 end: segment.start + ((index + 1) * wordDuration)
//                             };
//                         });
//                         words.push(...segmentWords);
//                     } else {
//                         console.warn('No text or time info in segment:', segment);
//                     }
//                 });
//
//                 console.log('Words:', words);  // Debug: Log the words array
//
//                 words.forEach(word => {
//                     if (word && word.text) {
//                         const span = document.createElement('span');
//                         span.textContent = word.text + ' ';
//                         span.dataset.start = word.start;
//                         span.dataset.end = word.end;
//                         transcriptionDiv.appendChild(span);
//                     } else {
//                         console.warn('Invalid word:', word);
//                     }
//                 });
//
//                 audioPlayer.addEventListener('timeupdate', function() {
//                     const currentTime = audioPlayer.currentTime;
//                     console.log('Current Time:', currentTime);  // Debug: Log the current playback time
//
//                     let highlightFound = false;
//                     document.querySelectorAll('#transcription span').forEach(span => {
//                         const start = parseFloat(span.dataset.start);
//                         const end = parseFloat(span.dataset.end);
//
//                         if (currentTime >= start && currentTime <= end) {
//                             span.classList.add('highlight');
//                             highlightFound = true;
//                         } else {
//                             span.classList.remove('highlight');
//                         }
//                     });
//
//                     // Handle instrumental parts by checking if any span is highlighted
//                     if (!highlightFound) {
//                         document.querySelectorAll('#transcription span').forEach(span => {
//                             span.classList.remove('highlight');
//                         });
//                     }
//                 });
//             } else {
//                 console.error('No segments found in the response:', data);
//             }
//         })
//         .catch(error => {
//             console.error('Error:', error);  // Debug: Log any errors
//         });
//     });
// });

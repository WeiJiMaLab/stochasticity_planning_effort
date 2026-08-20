
var accumulator = true

function question(text, options){
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
        <div class="layout comprehension">
            <p>${text}</p>
        </div>
        `,
        choices: options, 
        data:{
            task: 'comprehension',
        },
        button_html: `<div><button class="jspsych-btn">%choice%</button></div>`
    }
}

function response(options, correct, description){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function(){
            var last_trial = jsPsych.data.getLastTrialData().trials
            var is_correct = last_trial[0].response == correct
            var is_correct_text = `<h2>That was incorrect.</h2><br/>`
        
            if (accumulator){
                accumulator = accumulator && is_correct            
            }

            if (is_correct){
                is_correct_text = `
                <h2>That was correct!</h2><br/>
                <p>${description}
                </p>
                `
            }

            return `<div class="layout comprehension">
            ${is_correct_text}
            <br/><br/>
            <p>Press the <b>spacebar</b> to continue.</p>
            </div>`
        }, 
        choices: [" "]
    }
}

function intro(){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function(){
            //reset the accumulator
            accumulator = true

            return `<div class="layout comprehension">
            <h2>This is a Comprehension Check</h2>
            <p>The following questions will test your comprehension on the 
            previous slides. <br/><br/> You must get all of them correct to proceed, 
            otherwise you will be asked to re-read the instruction slides. <br/><br/>
            <b>Press the <b>spacebar</b> to continue.</b> </p>
            </div>`
        },
        choices: [" "]
    }
}

function outro(){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function(){

            if (accumulator){
                return `<div class="layout comprehension">
                <h2>Wonderful! You have passed the comprehension check.</h2>
                <p>Press the <b>spacebar</b> to continue.</p>
                </div>`
            }
            else{
                return `<div class="layout comprehension">
                <h2>You have not passed the comprehension check.</h2>
                <p>Please read through the informational slides again. 
                Press the <b>spacebar</b> to continue.</p>
                </div>`
            }
        },
        choices: [" "]

    }
}

export function comprehension_set(instruction, set){
    var nodes = []
    nodes.push(instruction)

    nodes.push(intro())

    set.map(s =>{
        nodes.push(question(s.text, s.options, s.answer))
        nodes.push(response(s.options, s.answer, s.desc))
    })

    nodes.push(outro())

    return {
        timeline: nodes,
        loop_function: function(){
        if(accumulator){
            return false;
        } else {
            return true;
        }
        }
    }
}

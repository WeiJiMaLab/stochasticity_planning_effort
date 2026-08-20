import {config, push, set} from './utils.js'

export function survey(prolific_id){
    var timeline = []
    timeline.push(consent(prolific_id))
    timeline.push(demographics())
    return {timeline: timeline}
}

function consent(prolific_id){
    return {
        type: jsPsychSurvey,
        pages: 
        [
            [
                {
                    type: 'html',
                    prompt: `Thank you for participating in this study. 
                    To begin, we will have you sign a consent form and answer a 
                    few short questions. When you complete the short survey, you will be 
                    asked to play a series of games. Completion of the study involves
                    both the completion of the survey and the games.`
                },
            ],
            [
                {
                    type: 'html',
                    prompt: `You have been invited to take part in a research study to learn more about the effect of uncertainty on planning. Their faculty sponsor for this study is Professor Wei Ji Ma, Arts & Science, New York University. If you agree to be in this study, you will be asked to do the following:<br/><br/>
                    <li>Complete a questionnaire about your background
                    <li>Play a computer game that asks you to collect the maximum amount of treasure<br/><br/>
                    
                    Participation in this study will involve approximately one hour. If you fail to complete the study within 140 minutes you will be timed out and not be paid. There are no known risks associated with your participation in this research beyond those of everyday life. Although you will receive no direct benefits, this research may help the investigator understand the effect of uncertainty on planning.                    

                    Confidentiality of your research records will be strictly maintained by the researcher. The only identifying information collected will be your Prolific ID. The results of this study may be published, but you will not be identified in these publications. Study personnel will make every effort to minimize the risk of data breaches and participant information loss. Information not containing identifiers may be used in future research, shared with other researchers, or placed in a data repository without your additional consent.<br/><br/>
                    
                    The base pay for this study will be $8 for completion of the experiment, with possible performance-based bonuses. If you fail the attention checks or withdraw early from the study you will not be compensated.<br/><br/>
                    
                    Participation in this study is voluntary. You may refuse to participate or withdraw at any time without penalty. For interviews, questionnaires, or surveys, you have the right to skip or not answer any questions you prefer not to answer. If there is anything about the study or your participation that is unclear or that you do not understand, if you have questions or wish to report a research-related problem, you may contact Wei Ji Ma at weijima@nyu.edu.<br/><br/>
                    
                    For questions about your rights as a research participant, you may contact the University Committee on Activities Involving Human Subjects (UCAIHS), New York University, 665 Broadway, Suite 804, New York, New York, 10012, at ask.humansubjects@nyu.edu or (212) 998-4808. Please reference the study #(IRB-FY2023-7039) when contacting the IRB (UCAIHS).`,
                },
                {
                    type: 'multi-choice', 
                    prompt: "Please read the statement above and indicate whether you consent to participating in this study.", 
                    options: ['I consent to participating in this study.', 'I do not consent.'],
                    name: 'consent', 
                    required: true, 
                },
                {
                    type: 'html', 
                    prompt: `
                    We have provided your Prolific ID here for reference: <br/><b>${prolific_id}</b><br/>
                    You may copy and paste the text into the box below. <b>Please make sure it matches exactly.</b>`
                },
                {
                    type: 'text', 
                    prompt: "Please confirm your consent to participate by signing here with your Prolific ID.",
                    name: 'signature', 
                }
            ]
        ],
        button_label_next: 'Continue',
        button_label_back: 'Previous',
        button_label_finish: 'Continue',
        show_question_numbers: 'onPage', 
        on_finish: (data) =>{
            if (data.response.consent != "I consent to participating in this study."){
                jsPsych.endExperiment('<p>You have selected "I do not consent". The session is now finished.</p>');
            }
            else if (data.response.signature.trim() != prolific_id){
                jsPsych.endExperiment('<p>Your signature does not match your Prolific ID. The session is now finished.</p>');
            }
            else{
                data.type = "survey"
                set("surveydata/0", data.response)
                push("data/", data)
            }
        }
    };
}

function demographics(){
    return {
        type: jsPsychSurvey,
        pages: 
        [
            [
                {
                    type: 'html',
                    prompt: `Please fill out the following demographic information.`
                },
                {
                    type: 'text', 
                    prompt: `Please fill in your age in years.`, 
                    input_type: 'number',
                    required: true,
                },
                {
                    type: 'multi-choice', 
                    prompt: "How would you describe yourself?", 
                    options: ['American Indian / Alaskan Native', 'Asian', 'Native Hawaiian or Other Pacific Islander', 'Black or African American', 'White', 'Other', 'Prefer Not to Answer'],
                    name: 'race', 
                    required: true, 
                },
                {
                    type: 'text', 
                    prompt: 'If other, please specify.',
                    name: "if_other", 
                },
                {
                    type: 'multi-choice', 
                    prompt: "Are you of Hispanic, Latino, or of Spanish origin?",
                    options: ["Yes", "No", "Prefer Not to Answer"],
                    name: "ethnicity", 
                    required: true
                },
                {
                    type: 'multi-choice', 
                    prompt: "What is your gender?",
                    options: ["Male", "Female", "Nonbinary", "Prefer Not to Answer"],
                    name: 'gender', 
                    required: true
                }
            ],
        ],
        button_label_next: 'Continue',
        button_label_back: 'Previous',
        button_label_finish: 'Continue',
        on_finish: (data) => {
            data.type = "survey"
            set("surveydata/1/", data.response)
            push("data/", data)
        }
    };
}

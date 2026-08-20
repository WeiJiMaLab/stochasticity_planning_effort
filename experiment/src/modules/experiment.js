import {game_nodes, CustomGame, game_boundary_check} from './game.js';
import {comprehension_set} from './comprehension.js';
import {shuffle, pressY, 
    pressSpacebar, instruction_content, instruction_node, 
    pascal, config, push, set, detect_offline, enter_fullscreen,
    detect_existing_participant, browser_check,
    detect_fullscreen} from './utils.js';
import {Reliability, Controllability, Volatility} from './stochtypes.js'
import {survey} from './survey.js'

/**
 * Main body of the experiment
 * 
 * @param {List} timeline the timeline from jsPsych 
 */
export function experiment(timeline){
    // Demo mode must be resolved before any Firebase-backed call: set() and
    // detect_existing_participant() are no-ops only when config.demo is true.
    config.demo = ["demoR", "demoV", "demoC", "demoT"].includes(config.stoch_type)

    detect_existing_participant()
    
    set("stoch_type/", config.stoch_type)
    set("prolific_id/", config.prolific_id)

    var num_games_per_block = 15
    var num_conditions = 5

    if (config.stoch_type == "R2024"){var game = new Reliability(num_conditions)}
    else if (config.stoch_type == "V2024"){var game = new Volatility(num_conditions)}
    else if (config.stoch_type == "T2024"){var game = new Controllability(num_conditions)}
    else if (config.stoch_type == "demoR"){
        var game = new Reliability(num_conditions)
    }
    else if (config.stoch_type == "demoV"){
        var game = new Volatility(num_conditions)
    }
    // demoT is the former name for the controllability demo; kept as an alias.
    else if (config.stoch_type == "demoC" || config.stoch_type == "demoT"){
        var game = new Controllability(num_conditions)
    }
    

    else{throw new Error("StochType Error: experiment type not defined")}

    if (config.demo){
        // ?skip=1 drops the viewer straight into the games. Everything skipped
        // here is either a screen-compatibility gate or a participant-training
        // step (instructions, comprehension check, practice games) that a demo
        // visitor does not need.
        if (!config.skip_intro){
            // timeline.push(browserCheck())
            timeline.push(game_boundary_check())
            // timeline.push(survey(config.prolific_id))
            // timeline.push(welcome())
            timeline.push(instructions_generic())
            timeline.push(game.instruction_blocks())
        }
        timeline.push(game.game_blocks(1))
    }
    else{
        timeline.push(enter_fullscreen())
        timeline.push(browser_check())
        timeline.push(game_boundary_check())
        timeline.push(survey(config.prolific_id))
        timeline.push(welcome())
        timeline.push(instructions_generic())
        timeline.push(game.instruction_blocks())
        timeline.push(game.game_blocks(num_games_per_block))
        timeline.push(submit())
    }
    
    return timeline
}

// ****************************************************************** HELPER FUNCTIONS BELOW THIS LINE ******************************************************************
/**
 * Uploads data to Firebase
 * 
 * @param {Number} id the user id
 * @param {object} data the data to be uploaded (in JSON Object form)
 */
export function uploadData(id, data){
    config.set(config.ref(config.database, config.stoch_type + "/" + id), data);
}

function welcome(){
    var timeline = []

    timeline.push(
        pressSpacebar(`
        <h1>Welcome!</h1>
        <p>
        Prolific id=${config.prolific_id}<br/>
        Please confirm that we have the correct Prolific ID 
        Shown above and press the <b>spacebar</b> to continue</p>
    `)
    )

    timeline.push(
        pressSpacebar(`
        <div class="comprehension">
            <h1>Information</h1>
            <p>
                You may take as much time as you like on this experiment. 
                The estimated total time is approximately one hour. In this experiment
                you will be asked to play a series of games.
                You will be paid a base pay of $8 for completing this experiment. <br/><br/>

                You will receive a <b>bonus payment based on your performance on these games.</b> 
                The bonus will be calculated from a random sample of the games you play.
                <br/><br/>

                Note: Because the bonus is calculated separately, please allow some time after the study to receive
                the additional bonus.
            </a>
            <br/><br/>
            Press the <b>spacebar</b> to continue</p>
        </div>
        `)
    )

    timeline.push(
        pressSpacebar(`
        <div class="comprehension">
            <h1>Information</h1>
            <p>
                <b>Important:</b> If you drop out early or end the experiment before it is complete, you will not be paid. 
                In addition, there will be attention checks in the experiment. If you fail two of them, 
                the experiment will end and you will not be paid.
            </a>
            <br/><br/>
            Press the <b>spacebar</b> to continue</p>
        </div>
        `)
    )

    return {timeline: timeline}
}



function instructions_generic(){
    var instruction_stack = []
    instruction_stack.push(
        instruction_content(
            `
            <h1>Please read these instructions carefully.
            </h1>
            `, 
            'closed.png'
        )
    )
    instruction_stack.push(
        instruction_content(
            `
            <div> 
            <h2>Welcome, explorer!</h2>
            You are an explorer seeking treasure. Luckily, you have a 
            treasure map to guide your hunt.<br/>

            <br/><br/>
            <b>Your goal is to get the most points possible.</b>
            </div>
            `, 
            'closed.png'
        )
    )
    instruction_stack.push(
        instruction_content(
            `
            <h2>
            Treasure chests
            </h2>
            Each treasure chest is labeled with a number from 1 to 9.
            This number tells you how many points you get. 
            
            <br/><br/> <b>The higher the number, the better.</b>
            `, 
            'intro0.png'
        )
    )

    instruction_stack.push(
        instruction_content(
            `
            <h2>
            How to move
            </h2>
            At each step, you can move either
            down-and-left or down-and-right from your current spot.
            <br/><br/>
            You will press the <b>F key to move left</b>  and <b>J key to move right</b> .

            <br/><br/> <b>Your goal is to collect as many points as possible as you move down the board.</b>
            `, 
            'howtomove.png'
        )
    )

    instruction_stack.push(
        instruction_content(
            `
            <h2>
            Let's practice!
            </h2>
            Now, you will play a mini-version of the game. 
            <b/>This is a practice round and not a real game.</b> <br/><br/>
            Press <b>Next</b> to begin.
            `
        )
    )

    var comprehension = comprehension_set(
        {
            timeline: [
                detect_fullscreen(),
                instruction_node(instruction_stack),
                practice(),
            ]
        }
        , 
        [
            {
                text: "<h3>What is the goal of the game?</h3>", 
                options: [
                    "Collect the most points",
                    "Collect the least points",
                    "Finish the game as fast as possible"
                ], 
                desc: "The goal of the game is to collect the most points.",
                answer: 0
            },
            {
                text: "<h3>Which of the following statements is correct?</h3>", 
                options: [
                    "The J key is for moving left and the F key is for moving right",
                    "The arrow keys are used to navigate the board",
                    "The F key is for moving left and the J key is for moving right",
                ], 
                desc: "The F key is for moving left and the J key is for moving right.",
                answer: 2
            },
            {
                text: `<div class="img-container"><img src='images/reachable.png' height=300/></div><br/><br/>
                       <h3>Which treasure chests can still be reached?</h3>`
                , 
                options: [
                    "A and B",
                    "B and C",
                    "C and D",
                    "All of the above"
                ], 
                desc: "You can only reach treasure chests C and D from the pin.",
                answer: 2
            },
            {
                text: `<div class="img-container"><img src='images/score.png' height=300/></div> <br/> <br/> 
                       <h3>What score is associated with this game so far?</h3>`
                , 
                options: [
                    "3",
                    "9",
                    "11",
                    "Impossible to say"
                ], 
                desc: "The score of the game is 3 + 4 + 2 = 9",
                answer: 1
            },
        ]
    )
    return comprehension
}
function practice(){
    return game_nodes(new CustomGame("practice", 4, [0.0, 0.0, 0.0], false), `<p>Practice</p>`)
}

function submit(){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus:`
        <div>
            <p>Thank you for taking part in the study!<p/>
            <a href='https://app.prolific.co/submissions/complete?cc=CEIGQHZK'> [Important: Click here to complete the study and be redirected to Prolific] </a>
            </h2>
        </div>
        `,
        stimuli: [" "],
        choices: ['none'],
        timing_response: .001,
        timing_post_trial: 0,
        on_load: function() {
            var data = jsPsych.data.get().trials

            // number of games to sample randomly
            var n_games = 15

            // why do we randomly sample? 
            // to avoid people satisficing on the games
            var totals_ = []

            data.filter(e => e.type == "game_end").forEach(
                ({game, rt, trial_index, trials}) => {
                    if (!game.name.includes("practice")){
                        totals_.push(game.total - game.baseline)
                    }
                }
            )

            // randomly sample games
            var totals = shuffle(totals_)
            var earnings = totals.slice(0, n_games).reduce((a, b) => a + b, 0)
            earnings = Math.max((earnings / n_games), 0)

            var attn_checks = data.filter(e => e.type == 'attn_check')
            var attn_check_fails = 0
            if (attn_checks.length > 0){
                attn_check_fails = attn_checks.at(-1).total_failed
            }

            set("attn_check_fails/", attn_check_fails)
            set("earnings/", earnings)
        }
    }
}



export var config = {}

var vw = window.screen.availWidth;
var vh = window.screen.availHeight;
const MIN_INNER_WIDTH = 1000
const MIN_INNER_HEIGHT = 700

/**
 * Given an array, shuffles it
 * 
 * @param {List} array 
 * @return {List} the shuffled array
 */
export function shuffle(array) {
    let currentIndex = array.length,  randomIndex;
  
    // While there remain elements to shuffle.
    while (currentIndex != 0) {
  
      // Pick a remaining element.
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
    return array;
}

/**
 * Checks to see if participant is online
 */
export function detect_offline(){
    if (!config.demo){
        const connectedRef = config.ref(config.database, ".info/connected");
        config.on_value(connectedRef, (snap) => {
        if (snap.val() === true) {
            //
        } else {
            alert("Error code: CONNECTION ERROR.\n \
                    It appears that your internet connection is unstable.\
                    Please copy the error code above and contact us.\
                    Please close the experiment now.")
        }
        });
    }
}

export function browser_check(){
    return {
        type: jsPsychBrowserCheck,
        inclusion_function: (data) => {
          // add a detector for if the participant is offline - will report whenever the connectivity changes
          detect_offline()
          return data.browser == 'chrome' && data.mobile === false && data.width >= MIN_INNER_WIDTH && data.height >= MIN_INNER_HEIGHT
        },
        exclusion_message: (data) => {
          if(data.mobile){
            jsPsych.endExperiment('Sorry, this experiment is not compatible with mobile and tablet devices. Please close this window and reopen this experiment on a desktop or laptop.');
          } else if(data.browser !== 'chrome'){
            jsPsych.endExperiment('Sorry, this experiment works on Chrome only. Please close this window and reopen this experiment in chrome.');
          }
          else{
            jsPsych.endExperiment('Sorry, this experiment is not compatible with this screen size or aspect ratio. We are sorry for the inconvenience. You can close this browser window now.');  
          }
        },
    };
}

export function enter_fullscreen(){
    return {
        type: jsPsychFullscreen,
        message: `<p>
        <b>
        Please ensure that you are using the latest version of Chrome. If not, please
        reopen this experiment in a new Chrome window.
        </b>
        <br/><br/>
        By pressing the button below, you will enter full screen mode.
        <br/><br/>
        Please remain in full screen mode - do not exit or press
        the escape key unless you are finished or wish to end the experiment. Thank you!
        </p> 
        `,
        fullscreen_mode: true, 
    }
}

export function detect_fullscreen(){
    var reenter_fullscreen = {
        type: jsPsychFullscreen,
        message: `<p>
        <b>
        Uh oh - It appears that the experiment has exited full screen mode!
        </b>
        <br/><br/>
        By pressing the button below, you will re-enter full screen mode.
        Please remain in full screen mode - do not exit or press
        the escape key unless you are finished or wish to end the experiment. Thank you!
        </p> 
        `,
        fullscreen_mode: true, 
    }

    var if_node = {
        timeline: [reenter_fullscreen, browser_check()],
        conditional_function: function(){
            if(document.fullscreenElement && window.innerWidth >= MIN_INNER_WIDTH && window.innerHeight >= MIN_INNER_HEIGHT){ 
                return false 
            } else {
                return true
            }
        }
    }
    return {timeline: [if_node]}
}

/**
 * Detects participant refresh and overwriting data
 */
export function detect_existing_participant(){
    if (!config.demo){
        config.get(config.ref(config.database, config.stoch_type + "/" + config.prolific_id + "/" + "data")).then(
            (snapshot) => {
                if (snapshot.exists()) {
                    alert("Error code: PARTICIPANT DATA EXISTS (POSSIBLE REFRESH) ERROR.\n\
                    There has been an error in the experiment.\
                    Please copy the error code above and contact us.\
                    Please close the experiment now.")
                    if(config.prolific_id.includes("pilot_jordan")){

                    }
                    else{
                        jsPsych.endExperiment("The experiment ended because of an error. Please close this window and contact us.")
                    }

                } else {
                    // no duplications
                }
            }
        )
    }
}

/**
 * Pushes data onto Firebase
 * 
 * @param {*} prefix 
 * @param {*} data 
 */
export function push(prefix, data){
    if (config.demo){
        console.log(data)
    }
    if (!config.demo){
        var error_id = prefix + config.counter + "/" + data.internal_node_id

        config.set(config.ref(config.database, config.stoch_type + "/" + config.prolific_id + "/" + prefix + config.counter + "/"), data).then(() => {
            // Data saved successfully!
        }).catch((error) => {
            alert("Error code: SET ERROR." + error_id + "\n\
                    There has been an error in the experiment.\
                    Please copy the error code above and contact us.\
                    Please close the experiment now.")
            jsPsych.endExperiment("The experiment ended because of an error. Please close this window and contact us.")

        });

        config.get(config.ref(config.database, config.stoch_type + "/" + config.prolific_id + "/" + prefix + config.counter + "/"), data).then(
            (snapshot) => {
                if (snapshot.exists()) {
                    //
                } else {
                    alert("Error code: MISSING DATA." + error_id + "\n\
                    There has been an error in the experiment.\
                    Please copy the error code above and contact us.\
                    Please close the experiment now.")
                    jsPsych.endExperiment("The experiment ended because of an error. Please close this window and contact us.")
                }
        }).catch((error) => {
            alert("Error code: GET ERROR." + error_id + "\n\
                    There has been an error in the experiment.\
                    Please copy the error code above and contact us.\
                    Please close the experiment now.")
            jsPsych.endExperiment("The experiment ended because of an error. Please close this window and contact us.")
        });

        config.counter = config.counter + 1
    }
}

/**
 * Sets data on Firebase
 * 
 * @param {*} prefix 
 * @param {*} data 
 */
export function set(prefix, data){
    if (config.demo){
        console.log(data)
    }
    if (!config.demo){
        config.set(config.ref(config.database, config.stoch_type + "/" + config.prolific_id + "/" + prefix), data).then(() => {
            // Data saved successfully!
        }).catch((error) => {
            alert("Saving data failed. Please contact us and close the experiment.")
        });
    }
}

/**
 * Creates an integer range array from start ... stop
 * with step intervals
 * @param {Number} start 
 * @param {Number} stop 
 * @param {Number} step 
 */
export function range(start, stop, step){
    return Array.from(
        { length: (stop - 1 - start) / step + 1 },
        (value, index) => start + index * step);
}

/**
 * Given a string formatted in HTML, creates a timeline node
 * for jsPsych
 * 
 * @param {String} html an input formatted in HTML
 * @return {object} a jsPsychHtmlKeyboardResponse node
 */
export function node(html){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
        <div class="layout">
        ${html}
        </div>`,
        on_finish: (data) => push("data/", data)
    }
}

/**
 * Creates a node where you press Y to advance
 * @param {String} html 
 */
export function pressY(html){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
        <div class="layout">
        ${html}
        </div>`
        ,
        choices: ["y"],
        on_finish: (data) => push("data/", data)
    }
}


export function pressKey(html, key){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
        <div class="layout">
        ${html}
        </div>`
        ,
        choices: [key],
        on_finish: (data) => push("data/", data)
    }
}

/**
 * Creates a node where you press spacebar to advance
 * @param {String} html 
 */
export function pressSpacebar(html){
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
        <div class="layout">
        ${html}
        </div>`
        ,
        choices: [" "],
        on_finish: (data) => push("data/", data)
    }
}

/**
 * Creates an instruction node
 * @param {String} html html code
 * @param {String} image optional image path
 */
export function instruction_content(html, image){
    if (image){
        return`
        <div class = "instructions">
        <div class = "row">
        <div class= "col-md-6 instructionpanel">
            ${html}
        </div>
        <div class= "col-md-6">
                <img id="instructionCanvas" src = "./images/${image}"/>
        </div>
        </div>
        </div>`
    }
    else{
        return `
        <div class = "instructions lonetext">
            ${html}
        </div>
        `
    }  
}

/**
 * Creates a navigation panel for instruction nodes
 * @param {List} instruction_stack a list of instruction_content nodes
 */
export function instruction_node(instruction_stack){
    var node = {
        type: jsPsychInstructions,
        pages: instruction_stack,
        show_clickable_nav: true,
        allow_keys: false,
        on_finish: (data) => push("data/", data)
    }
    return node
}

/**
 * Creates a deep copy of an array
 * @param {Object} object to be deepcopied
 */
export function deepcopy(a){
    return JSON.parse(JSON.stringify(a))
}

/**
 * Creates a unique identifier using UUID method
 */
export function uuid(){
    var S4 = function() {
        return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
    };
    return (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4());
}

/**
 * Creates a unique ID with a prefix
 * @return {String} a unique string
 */
export function makeId(prefix) {
    if (!prefix){
        prefix = ""
    }
    
    return prefix + "-" + uuid();
}

export function randint(lo, hi){
    return Math.floor(Math.random() * (hi - lo)) + lo
}

/**
 * Creates a nxn square matrix of zeros
 * @param {Number} n the width and length of the square matrix
 */
export function zeros(dims){
    var mtx = []
    if (dims.length == 1){
        for (var row = 0; row < dims[0]; row++){
            mtx.push(0)
        }
    }
    else if (dims.length == 2){
        for (var row = 0; row < dims[0]; row++){
            var r = []
            for (var col = 0; col < dims[1]; col++){
                r.push(0)
            }
            mtx.push(r)
        }
    }
    else{
        for (var row = 0; row < dims[0]; row++){
            var r = []
            for (var col = 0; col < dims[1]; col++){
                var c = []
                for (var k = 0; k < dims[2]; k++){
                    c.push(0)
                }
                r.push(c)
            }
            mtx.push(r)
        }
    }
    
    return mtx
}

export function pascal(n, normalize){
    var p = zeros([n, n])

    for (var row = 0; row < n; row ++){
        for (var col = 0; col <= row; col ++){
            if (col == 0 || row == col){
                p[row][col] = 1
            }
            else{
                p[row][col] = (p[row - 1][col - 1] + p[row - 1][col])
            }
        }
    }

    if (normalize){
        for (var row = 0; row < n; row ++){
            var rowsum = 0
            for (var col= 0; col <= row; col ++){
                p[row][col] = p[row][col] / Math.pow(2, row)
                rowsum += p[row][col]
            }
        }
    }

    return p
}


export function dotprod(a, b){
    var prod = 0;

    for (var row = 0; row < a.length; row ++){
        for (var col = 0; col < a[0].length; col ++){
            prod += a[row][col] * b[row][col]
        }
    }

    return prod
}

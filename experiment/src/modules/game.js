import {push, set, makeId, randint,
   zeros, deepcopy, pressSpacebar, 
   dotprod, pascal, browser_check, 
   detect_fullscreen} from './utils.js';


// load the treasure chest images for later use
let id = null;
const closed = new Image()
const openBack = new Image()
const openFront = new Image()
const youAreHere = new Image()
const coinDefault = new Image()
const coinVisit = new Image()
const closedMagic = new Image()
closed.src = 'images/closed.png'
openBack.src = 'images/open-back.png'
openFront.src = 'images/open-front.png'
youAreHere.src = "images/youarehere.png"
coinDefault.src = 'images/coin_default.png'
coinVisit.src = 'images/coin_visit.png'
closedMagic.src = 'images/closed_magic.png'

// These default width and height values
// are selected so that the entire board fits on the screen
// DO NOT ALTER THEM WITHOUT CHECKING
var canvasdefaultwidth = 1500;
var canvasdefaultheight = 900;

window.onload = function() {
  var width = window.innerWidth
  var height = window.innerHeight
  resize(width, height);
};

window.onresize = function() {
  var width = window.innerWidth
  var height = window.innerHeight
  resize(width, height); 
};

function resize(w, h){
  var scale_factor = Math.min(w, 1.7 * h) / 1500
  scale_factor = Math.min(1, scale_factor)
  document.documentElement.style.setProperty(`--scale`, `${scale_factor * 100}%`);
}

// size of the treasure chest image the screen
const size_default = 100
const size_visit = 130
const coinsize = 45

// color of the coins
const coin_visit = [224, 178, 9]

export class Game {
  /**
   * A object-oriented class Game which is in charge of
   * the visualization of a board arrangement, as described
   * by JSON objects
   */
  constructor(name, n, p_unreliable, p_volatile, p_transition, show_e_zone) {
    this.name = name
    this.n = n
    this.p_unreliable = p_unreliable
    this.p_volatile = p_volatile
    this.p_transition = p_transition
    this.show_e_zone = show_e_zone

    // things above this line don't change
    this.reset()
  }

  reset(){   
    // regenerates board configuration from scratch
    var board_vars = make_boards(this.n, this.p_unreliable, this.p_volatile, this.p_transition)
    this.boards = board_vars[0]
    this.oracle = board_vars[1]
    this.is_unreliable = board_vars[2]
    this.is_volatile = board_vars[3]
    this.is_transition = board_vars[4]
    this.name = this.name + "-mod"

    var pasc = pascal(this.n, true)
    this.baseline = dotprod(pasc, this.oracle)

    this.done = false
    this.actions = []
    this.path = []
    this.tuplepath = []
    this.r = 0
    this.c = 0

    this.path.push(this.r + "," + this.c)
    this.tuplepath.push([this.r, this.c])

    this.total = this.oracle[this.r][this.c]
  }

  /**
   * Alters the internal state of the game depending on the action taken
   * 
   * @param {Boolean} moveLeft - true if the left key (F) is pressed
   * @returns {Boolean} - true if the move is valid
   */
  act(moveLeft){

    if (this.is_transition[this.r]){
      if (moveLeft){
        this.c = this.c + 1
        this.actions.push(moveLeft)
      }
      else{
        this.actions.push(moveLeft)
      }
    }
    else{
      if (moveLeft){
        this.actions.push(moveLeft)
      }
      else{
        this.c = this.c + 1
        this.actions.push(moveLeft)
      }
    }

    // increment the row
    this.r = this.r + 1

    // get the reward from the true value of the state 
    this.total += this.oracle[this.r][this.c]
    // add the current position onto the path
    this.path.push(this.r + "," + this.c)
    this.tuplepath.push([this.r, this.c])

    // should always match what the player is seeing
    this.tuplepath.forEach(e => 
      {
        this.boards[this.r][e[0]][e[1]] = this.oracle[e[0]][e[1]]
      }
    )

    // if we have reached the last row, we are done
    if (this.r == this.n - 1){
      this.done = true
    }
    return true
  }

  /**
   * Takes an HTMLCanvas element and draws the board
   * 
   * @param {HTMLCanvasElement} canvas - the canvas where you want to draw the board
   */
  draw(canvas){
    var context = canvas.getContext('2d')
    clearInterval(id);
    let counter = 0;
    let g = this;

    id = setInterval(function() {
      if (counter > 300) {
        clearInterval(id);
      } else {
        counter++;
        anim(g, counter);
      }
    }, 5);

    context.scale(3,3);
  }
}

export class CustomGame extends Game{
  constructor(name, n, probabilities, show_e_zone){
    var [p_unreliable, p_volatile, p_transition] = probabilities

    if (!name){
      name = "game"
    }

    super(makeId(name), n, p_unreliable, p_volatile, p_transition, show_e_zone)
  }
}

/**
 * Creates an n x n board
 * @param {Number} n 
 */
export function initialize_game(n){
  var board = zeros([n, n])
  for (var row = 0; row < n; row ++){
    for (var col = 0; col <= row; col ++){
      board[row][col] = randint(1, 10)
    }
  }

  board[0][0] = 0
  return board
}

export function make_boards(n, p_unreliable, p_volatile, p_transition){
  var oracle = zeros([n, n])
  var boards = zeros([n, n, n])

  var is_unreliable = zeros([n, n])
  var is_volatile = zeros([n, n, n])
  var is_transition = zeros([n-1])
  

  boards[0] = initialize_game(n)

  // trial t, row r, column c
  // this for loop tracks a single element's evolution
  // through time t
  for (var r = 1; r < n; r ++){
    for (var c = 0; c <= r; c ++){
      for (var t = 1; t < n; t ++){

        // a volatile element will randomly change its value
        // elements in row r < t no longer change; only future
        // elements down the tree are eligible to be volatile
        if (Math.random() < p_volatile && r >= t){
          is_volatile[t][r][c] = true
          boards[t][r][c] = randint(1, 10)
        }

        // a non-volatile element will keep the previous value
        else{
          is_volatile[t][r][c] = false
          boards[t][r][c] = boards[t - 1][r][c]
        }
      }
      // at this point in the loop, a given element (r, c)
      // has set all the values across time t

      // the final value of element (r, c) is its true value
      oracle[r][c] = boards[n-1][r][c]

      if (Math.random() < p_unreliable){
        oracle[r][c] = randint(1, 10)
        is_unreliable[r][c] = true
      }
    }
    if (Math.random() < p_transition){
      is_transition[r-1] = true
    }
  }

  return [boards, oracle, is_unreliable, is_volatile, is_transition]
}

// ******************************************************************* GAME VISUALS   ****************************************************************
/**
 * Converts a (r, c) coordinate system to positions on an HTML canvas
 * 
 * @param {HTMLCanvasElement} canvas the canvas where the object will be drawn
 * @param {Number} r the row in the game matrix
 * @param {Number} c the column in the game matrix
 * @param {Number} offset an integer (pixel) offset
 */
export function get_coords(canvas, r, c, offset){
  if (!offset){
    offset = 0
  }
  const minY = 160
  const spacing = size_visit * 1.15
  const centerX = canvas.width / 2 / 3;
  const posX = centerX - spacing/2 * r + spacing * c + offset
  const posY = minY + r * (spacing/2 + 10)
  return [posX, posY]
}

export function draw_start_node(canvas, r, c, val, color){
  const context = canvas.getContext('2d');
  var [posX, posY] = get_coords(canvas, r, c)
  const radius = 40;

  context.beginPath();
  context.arc(posX, posY-10, radius, 0, 2 * Math.PI, false);
  var colorval = color //color.map(x => x * (1 - (val + 5) / 15) + 255 * ((val + 5) / 15))
  context.fillStyle = 'rgb('+ colorval.join() +')'
  context.fill();

  context.font = "22px sans-serif";
  context.fillStyle = "black";
  context.fillText(val, posX - 35, posY);
}

function anim(g, counter){
  var canvas = document.getElementById('gameCanvas')
  if(!canvas){
    return null;
  }
  var maxcount = 60
  var transition_time = 20

  const width = canvas.width; 
  const height = canvas.height;
  const ctx = canvas.getContext('2d');
  var [posX, posY] = get_coords(canvas, g.r, g.c)

  ctx.clearRect(0, 0, width, height); // clear canvas
  // ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  // ctx.strokeStyle = 'rgba(0, 153, 255, 0.4)';
  ctx.save();
  var [startX, startY] = get_coords(canvas, 0, 0)
  var startsize = 80
  ctx.translate(startX, startY)

  if (g.r == 0)
  {
    ctx.drawImage(coinVisit, -startsize/2, -startsize/2, startsize, startsize)
  }
  else{
    ctx.drawImage(coinDefault, -startsize/2, -startsize/2, startsize, startsize)
  }
  ctx.font = "23px sans-serif";
  ctx.textAlign = "center"
  ctx.fillText("START", 0, 8);
  ctx.restore();
  ctx.save();

  ctx.translate(posX-36/2, posY-90); 
  if (counter > transition_time){
    ctx.drawImage(youAreHere, 0, 0, 36, 50);
  }
  ctx.restore();
  ctx.save();

  if (g.show_e_zone && g.r==0){
    ctx.beginPath();
    ctx.lineWidth = 5
    ctx.setLineDash([10, 10]);
    ctx.moveTo(-width, get_coords(canvas, g.r, g.c)[1] + size_default / 2);
    ctx.lineTo(width, get_coords(canvas, g.r, g.c)[1]+ size_default / 2);
    ctx.stroke();
    ctx.restore();
    ctx.save();
  }
  
  var angle = counter; 
  if (counter < maxcount){

    // if (g.show_e_zone && g.r!=0){
    //   ctx.beginPath();
    //   ctx.lineWidth = 5
    //   ctx.setLineDash([10, 10]);
    //   ctx.moveTo(-width, get_coords(canvas, g.r, g.c)[1] + size_default / 2);
    //   ctx.lineTo(width, get_coords(canvas, g.r, g.c)[1]+ size_default / 2);
    //   ctx.stroke();
    //   ctx.restore();
    //   ctx.save();
    // }


    for(let r = 1; r< g.n; r++){
      for(let c = 0; c< r+1; c++){
        [posX, posY] = get_coords(canvas, r, c)
        var size = size_default
        ctx.translate(posX, posY)
        ctx.font = "30px sans-serif";
        ctx.fillStyle = "white"
        
        if (r == g.r && c == g.c){
          size = size_visit
          ctx.font = "33px sans-serif";
          ctx.beginPath();

          if (g.actions.length > 0){
            if      (g.actions[g.r - 1] == 1 && g.is_transition[g.r - 1]){
              if (counter < transition_time){
                ctx.strokeStyle = "rgba(" + [255 * counter / transition_time, 0, 0, 0.6].join(",") + ")"
              }
              else{
                ctx.strokeStyle = "rgba(255, 0, 0, 0.6)"
              }
              var [prevX, prevY] = get_coords(canvas, r-1, c-1)
              var [otherX,otherY] = get_coords(canvas, r, c-1)
            }
            else if (g.actions[g.r - 1] == 1 && g.is_transition[g.r - 1] == 0){
              ctx.strokeStyle = "rgba(0, 0, 0, 0.6)"
              var [prevX, prevY] = get_coords(canvas, r-1, c)
              var [otherX,otherY] = [posX, posY]
            }
            else if (g.actions[g.r - 1] == 0 && g.is_transition[g.r - 1]){
              if (counter < transition_time){
                ctx.strokeStyle = "rgba(" + [255 * counter / transition_time, 0, 0, 0.6].join(",") + ")"
              }
              else{
                ctx.strokeStyle = "rgba(255, 0, 0, 0.6)"
              }
              
              var [prevX, prevY] = get_coords(canvas, r-1, c)
              var [otherX,otherY] = get_coords(canvas, r, c+1)
            }
            else if (g.actions[g.r - 1] == 0 && g.is_transition[g.r - 1] == 0){
              ctx.strokeStyle = "rgba(0, 0, 0, 0.6)"
              var [prevX, prevY] = get_coords(canvas, r-1, c-1)
              var [otherX,otherY] =  [posX, posY]
            }
          }

          ctx.moveTo(prevX - posX,  prevY - posY);
          ctx.lineWidth = 10;
          if (counter < transition_time){
            var tr = (transition_time - counter) / (transition_time)
            ctx.lineTo((otherX - posX)*tr, 0)
          }
          else{
            ctx.lineTo(0, 0)
          }
          ctx.globalCompositeOperation = 'destination-over';
          ctx.stroke();
          ctx.globalCompositeOperation = 'source-over';
        }
        // shake if the element is volatile, or if it is unreliable AND currently being visited
        if (g.is_volatile[g.r][r][c] || r == g.r && c == g.c && g.is_unreliable[r][c]){
          ctx.rotate(Math.sin(angle / 5) / 10);
        }

        ctx.translate(-size/2, -size/2);
        
        if (g.path.includes(r + "," + c) && !(r == g.r && c == g.c)){
          ctx.drawImage(openBack, 0, 0, size, size)
          ctx.drawImage(coinDefault, size/2-coinsize/2, size/2-coinsize/2 - 35, coinsize, coinsize)
          ctx.fillStyle = "black"
          ctx.textAlign = "center"
          ctx.fillText(g.oracle[r][c], size/2, size/2 - 25);
          ctx.restore()
          ctx.save()
        }
        else{
          if (g.is_volatile[g.r][r][c] || r == g.r && c == g.c && g.is_unreliable[r][c]){
            ctx.drawImage(closedMagic, 0, 0, size, size);
          }
          else{
            ctx.drawImage(closed, 0, 0, size, size);
          }
          
          ctx.fillStyle = "white"
          ctx.textAlign = "center"
          ctx.fillText(g.boards[Math.max(0, g.r - 1)][r][c], size/2, size/2 + 30);

          // important! remember to restore the context
          ctx.restore();
          ctx.save();
        }
      }
    }
  }
  else{
    if (g.show_e_zone){
      ctx.beginPath();
      ctx.lineWidth = 5
      ctx.setLineDash([10, 10]);
      ctx.moveTo(-width, get_coords(canvas, g.r, g.c)[1] + size_default / 2);
      ctx.lineTo(width, get_coords(canvas, g.r, g.c)[1] + size_default / 2);
      ctx.stroke();
      ctx.restore();
      ctx.save();
    }

    var tx = (counter - 60)/2;
    if (tx < 0){
      tx = 0;
    }
    else if (tx > 40){
      tx = 40
    }

    for(let r = 1; r< g.n; r++){
      for(let c = 0; c< r+1; c++){
        [posX, posY] = get_coords(canvas, r, c)
        var size = size_default
        ctx.translate(posX, posY)
        ctx.font = "30px sans-serif";
        ctx.translate(-size/2, -size/2);
        ctx.textAlign = "center"

        if (g.path.includes(r + "," + c) && !(r == g.r && c == g.c)){
          ctx.drawImage(openBack, 0, 0, size, size)
          ctx.drawImage(coinDefault, size/2-coinsize/2, size/2-coinsize/2 - 35, coinsize, coinsize)
          ctx.fillStyle = "black"
          
          ctx.fillText(g.oracle[r][c], size/2, size/2 - 25);
          ctx.restore()
          ctx.save()
        }
        else{
          ctx.drawImage(closed, 0, 0, size, size);
          ctx.fillStyle = "white"
          ctx.fillText(g.boards[g.r][r][c], size/2, size/2 + 30);

          // important! remember to restore the context
          ctx.restore();
          ctx.save();
        }
      }
    }

    if (g.r > 0){
      [posX, posY] = get_coords(canvas, g.r, g.c)

      const radius = 25
      ctx.translate(posX - size_visit/2, posY - size_visit/2)
      ctx.drawImage(openBack, 0, 0, size_visit, size_visit)
      ctx.beginPath();
      ctx.arc(size_visit/2, size_visit/2 - 2 * tx, radius, 0, 2 * Math.PI, false);
      var colorval = coin_visit
      ctx.fillStyle = 'rgb('+ colorval.join() +')'
      ctx.fill();

      ctx.font = "33px sans-serif";
      ctx.textAlign = "center"
      ctx.fillStyle = "black"
      ctx.fillText(g.oracle[g.r][g.c], size_visit/2, size_visit/2 + 10 - 2 * tx);
      ctx.drawImage(openFront, 0, 0, size_visit, size_visit)

    }
  }
    
  ctx.restore();
}

// ******************************************************************* JSPSYCH FUNCTIONS ****************************************************************
export function game_boundary_check(){
    var screencheck1 = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function() {
          return `
          <div class="screencheck-instructions">
          <p>Before we begin, we need to make sure your screen is suitable for the experiment.<p>
          You should see a <span style="color:red">red number</span> on screen. Please use your keyboard to <b>press the number you see</b>. 
          If you cannot see any letters, press the <b>spacebar</b>.
          </div>
          <div id="screenTest">
          <canvas id="screenCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
          </div>
          `;
      },
      on_load: function(){
        var canvas = document.getElementById('screenCanvas')
        const context = canvas.getContext('2d');
        context.scale(3, 3)
        var [posX, posY] = get_coords(canvas, -1, -0.5)
        context.font = "40px sans-serif";
        context.fillStyle = "red";
        context.textAlign = "center"
        context.fillText("4", posX, posY);
      },
      on_finish: (data) => {
        if(jsPsych.pluginAPI.compareKeys(data.response, "4")){
        } 
        else{
          jsPsych.endExperiment('Sorry, this experiment is not compatible with this screen. We are sorry for the inconvenience. You can close this browser window now.');
        }
      }
    }
  
    var screencheck2 = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function() {
          return `
          <div class="screencheck-instructions">
          <p>Before we begin, we need to make sure your screen is suitable for the experiment.<p>
          You should see a <span style="color:red">red number</span> on screen. Please use your keyboard to <b>press the number you see</b>. 
          If you cannot see any letters, press the <b>spacebar</b>.
          </div>
          <div id="screenTest">
          <canvas id="screenCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
          </div>
          `;
      },
      on_load: function(){
        var canvas = document.getElementById('screenCanvas')
        const context = canvas.getContext('2d');
        context.scale(3, 3)
        var [posX, posY] = get_coords(canvas, 8, 0)
        context.font = "40px sans-serif";
        context.fillStyle = "red";
        context.textAlign = "center"
        context.fillText("9", posX, posY);
      },
      on_finish: (data) => {
        if(jsPsych.pluginAPI.compareKeys(data.response, "9")){
        } 
        else{
          jsPsych.endExperiment('Sorry, this experiment is not compatible with this screen. We are sorry for the inconvenience. You can close this browser window now.');
        }
      }
    }


    var screencheck3 = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function() {
          return `
          <div class="screencheck-instructions">
          <p>Before we begin, we need to make sure your screen is suitable for the experiment.<p>
          You should see a <span style="color:red">red number</span> on screen. Please use your keyboard to <b>press the number you see</b>. 
          If you cannot see any letters, press the <b>spacebar</b>.
          </div>
          <div id="screenTest">
          <canvas id="screenCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
          <div>
          `;
      },
      on_load: function(){
        var canvas = document.getElementById('screenCanvas')
        const context = canvas.getContext('2d');
        context.scale(3, 3)
        var [posX, posY] = get_coords(canvas, 8, 8)
        context.font = "40px sans-serif";
        context.fillStyle = "red";
        context.textAlign = "center"
        context.fillText("2", posX, posY);
      },
      on_finish: (data) => {
        if(jsPsych.pluginAPI.compareKeys(data.response, "2")){
        } 
        else{
          jsPsych.endExperiment('Sorry, this experiment is not compatible with this screen. We are sorry for the inconvenience. You can close this browser window now.');
        }
      }
    }

    var screencheck4 = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function() {
          return `
          <div class="screencheck-instructions">
          <p>Before we begin, we need to make sure your screen is suitable for the experiment.<p>
          You should see a <span style="color:red">red number</span> on screen. Please use your keyboard to <b>press the number you see</b>. 
          If you cannot see any letters, press the <b>spacebar</b>.
          </div>
          <div id="screenTest">
          <canvas id="screenCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
          <h3><br/></h3>
          <p><br/><p>
          <br/>
          <p style="color:red; font-size:24px">5</p>
          </div>
          `;
      },
      on_load: function(){
        var canvas = document.getElementById('screenCanvas')
        const context = canvas.getContext('2d');
        context.scale(3, 3)
      },
      on_finish: (data) => {
        if(jsPsych.pluginAPI.compareKeys(data.response, "5")){
        } 
        else{
          jsPsych.endExperiment('Sorry, this experiment is not compatible with this screen. We are sorry for the inconvenience. You can close this browser window now.');
        }
      }
    }

  return {timeline: [screencheck1, screencheck2, screencheck3, screencheck4]}
}



/**
 * Draws a game frame for a given step of the game
 * 
 * @param {Game} game 
 * @return {object} jsPsychHtmlKeyboardResponse timeline node
 */
export function game_nodes(game, header){
  if (!header){
    header = ''
  }

  var begin_node = pressSpacebar(
    `
    <img id = "instructionCanvas" src = "./images/readyposition.png"/> <br/><br/>
    Please place your hands in the position shown above, with your <b>index fingers on F and J</b>.<br/> 
    Press the <b>spacebar</b> to continue.
    `
  )

  var node = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function(){
        document.documentElement.style.setProperty(`--width`, `${canvasdefaultwidth}px`);
        document.documentElement.style.setProperty(`--height`, `${canvasdefaultheight}px`);
        return `
        <canvas id="gameCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
        <h3>Points: ${game.total}</h3>
        ${header}
        <p>Move Left or Right with "F" and "J"</p>
        `
      },
      choices: ['F', 'J'],
      data:{
          type: 'game',
          name: game.name,
      },
      on_load: function(){
          var canvas = document.getElementById('gameCanvas')
          game.draw(canvas)
      },
      on_finish: function(data){
          if(jsPsych.pluginAPI.compareKeys(data.response, "f")){
            data.valid = game.act(true)
            data.move = 0
          } else {
            data.valid = game.act(false)
            data.move = 1
          }

          push("data/", data)
      }
  }

  var loop_node = {
    timeline: [node],
    loop_function: function(){
        if(!game.done){
            return true;
        } 
        else {
            return false;
        }
      },
      on_finish: (data) => push("data/", data)
    }

  var debrief_node = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
        <canvas id="gameCanvas" width="${canvasdefaultwidth * 3}" height="${canvasdefaultheight * 3}"></canvas>
        <h3>Your Total: ${game.total}&emsp; Press the <b>spacebar</b> to Continue...</h3>
        `;
    },
    choices: [" "],
    data:{
      type: 'game_end',
    },
    on_load: function(){
      var canvas = document.getElementById('gameCanvas')
      game.draw(canvas)
    },
    on_finish: function(data){
      var trial = jsPsych.data.getLastTimelineData().filter({valid: true}).trials
      data.trials = trial.map(({rt, trial_index, move}) => ({rt, trial_index, move}))
      data.game = deepcopy(game)
      push("data/", data)
    }
  }

  var result = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
        <img id="imgstyle" src="./images/goodjob.png"/> <br/><br/>
        <h3>Good job! Your total was:</h3>
        <h1>${game.total}</h1>
        <p>Press the <b>spacebar</b> to Continue</p>
        `;
    },
    choices: [" "],
    on_finish: (data) => push("data/", data)
  }

  return {
    timeline: [detect_fullscreen(), begin_node, loop_node, debrief_node, result],
    on_timeline_start: function(){
      game.reset()
    }
  }
}


export function attention_nodes(show_e_zone){
  var game = new CustomGame("attn", 8, [0.0, 0.0, 0.0], show_e_zone)

  var begin_node = pressSpacebar(
    `
    <img id = "instructionCanvas" src = "./images/readyposition.png"/> <br/><br/>
    Please place your hands in the position shown above, with your <b>index fingers on F and J</b>.<br/> 
    Press the <b>spacebar</b> to continue.
    `
  )

  var node = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function(){
        document.documentElement.style.setProperty(`--width`, `${canvasdefaultwidth}px`);
        document.documentElement.style.setProperty(`--height`, `${canvasdefaultheight}px`);
        return `
        <canvas id="gameCanvas" width="${3 * canvasdefaultwidth}px" height="${3 * canvasdefaultheight}px" style={'overflow':'scroll !important'}></canvas>
        <h3>Points: ${game.total}</h3>
        <p>Important: Press the spacebar only.</p>
        `
      },
      choices: ['F', 'J', ' '],
      data:{
          type: 'attn',
          name: game.name,
      },
      on_load: function(){
          var canvas = document.getElementById('gameCanvas')
          game.draw(canvas)
      },
      on_finish: function(data){
          if(jsPsych.pluginAPI.compareKeys(data.response, "F")){
            data.valid = game.act(true)
            data.move = 0
          } 
          else if(jsPsych.pluginAPI.compareKeys(data.response, "J")){
            data.valid = game.act(false)
            data.move = 0
          }
          else {
            data.move = Math.random() > 0.5
            data.valid = game.act(data.move)
          }
          push("data/", data)
      }
  }

  var loop_node = {
    timeline: [node],
    loop_function: function(){
        if(!game.done){
            return true;
        } 
        else {
            return false;
        }
      },
    on_finish: (data) => push("data/", data)
  }
  var debrief_node = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        var trials = jsPsych.data.get().filter({type: 'game', valid: true});
        var rt = Math.round(trials.select('rt').mean());
        return `
        <canvas id="gameCanvas" width="${canvasdefaultwidth * 3}" height="${canvasdefaultheight * 3}"></canvas>
        <h3>Your Total: ${game.total}&emsp; Press the <b>spacebar</b> to Continue...</h3>
        `;
    },
    choices: [" "],
    data:{
      type: 'attn_end'
    },
    on_finish: function(data){
      var trial = jsPsych.data.getLastTimelineData().filter({valid:true}).trials
      var sum = trial.map(({response}) => {if (jsPsych.pluginAPI.compareKeys(response, " ")){return 1} else{return 0}}
      ).reduce((a, b) => a + b, 0)

      data.game = deepcopy(game)
      data.attn_check = true

        if (sum < data.game.n - 1){
          data.attn_check = false
        }
      push("data/", data)
      },
    on_load: function(){
      var canvas = document.getElementById('gameCanvas')
      game.draw(canvas)
    }
  }

  var result = {
    type: jsPsychHtmlKeyboardResponse,
    choices: [" "],
    stimulus: function(){
        var last_trial = jsPsych.data.getLastTrialData().trials
        var is_correct = last_trial[0].attn_check
        var is_correct_text = `<p>Sorry, you failed the attention check.</p>`

        if (is_correct){
            is_correct_text = `<p>Good job! You passed the attention check.</p>`
        }

        return `<div>
        ${is_correct_text}
        <p>Press the <b>spacebar</b> to continue.</p>
        </div>`},

    on_finish: function(data){
      var total_failed = jsPsych.data.get().filter({type: 'attn_end'}).trials.map(({attn_check}) => {return !attn_check}).reduce((a, b) => a + b, 0)
      data.total_failed = total_failed 
      data.type = "attn_check"
      if (total_failed >= 2){
        jsPsych.endExperiment('<p>You have failed two attention checks. The experiment is now finished.</p>');
      }
      push("data/", data)
    }
  }

  return {
    timeline: [detect_fullscreen(), begin_node, loop_node, debrief_node, result],
    on_timeline_start: function(){
      game.reset()
    }
  }
}


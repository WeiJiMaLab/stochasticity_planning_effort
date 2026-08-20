
import {Game, game_nodes, attention_nodes, CustomGame} from './game.js';
import {comprehension_set} from './comprehension.js';
import {shuffle, node, pressSpacebar, pressKey, pressY, instruction_content, instruction_node, deepcopy, range, config} from './utils.js';

class StochasticityType{
    /**
     * Constructor for StochasticityType abstract class
     * @param {Number} n_conditions number of conditions
     * @param {Number} base denominator for stochasticity level - e.g. 4 in "1 in 4" chance
     * @param {String} interp interpretation of the condition, e.g. "chance that a treasure chest is a mystery chest"
     */
    constructor(type, n_conditions, base, interp){
        if(this.constructor == StochasticityType){
            throw new Error("StochasticityType is an abstract class and cannot be created");
        }
        this.n_conditions = n_conditions
        this.base = base
        this.interp = interp
    }

    /**
     * A function to return a list of instructions
     */
    instructions(){
        throw new Error("NotImplementedError: function instructions not implemented")
    }

    /**
     * A function to return a list of comprehension checks
     */
    comprehension_check(){
        throw new Error("NotImplementedError: function comprehension_check not implemented")
    }

    /**
     * A function to return an attention check; if a participant
     * fails two of these the game automatically terminates
     */
    attention_check(){
        var timeline = []
        timeline.push(pressSpacebar(`
        <div class="comprehension">
            <h1>This is an Attention Check.</h1>
            <p><i>Please follow these instructions carefully</i>. <br/><br/>
            If you fail 2 attention checks the experiment will end and you will not be paid.<br/><br/>
    
            <i>Instructions:</i> For each move in the next game, instead of pressing the F and J keys, 
            please press the <b>spacebar</b> for every step.
            
            <br/><br/>Press the <b>spacebar</b> to continue.
            </p>
        </div>
        `))
        timeline.push(this.create_attention_node())
        return {
            timeline: timeline
        }
    }

    /**
     * A function to create an attention node, overwritten in some classes
     */
    create_attention_node(){
        return attention_nodes(false)
    }

    /**
     * A function that creates a new CustomGame instance
     */
    create_game(){
        throw new Error("NotImplementedError: function create_game not implemented")
    }

    /**
     * Assembles instructions, practice, and comprehension check
     */
    instruction_blocks(){
        var instruction_stack = this.instructions()
        instruction_stack.push(
            instruction_content(
                `
                <h2>There are 5 game conditions.
                </h2>
                <br/>
                ` + range(0, this.n_conditions, 1).map(i => {
                    return `In <b>condition ${i + 1}</b> there is a 
                    <b>${i} in ${this.base} (${i / this.base * 100}%)</b>
                    ${this.interp}
                    `
                }).join(`<br/><br/>`)
            ), 

        )
        return comprehension_set(
            {timeline: [instruction_node(instruction_stack), this.practice_block()]}, 
            this.comprehension_check()
        )
    }

    /**
     * A function to create a practice block
     */
    practice_block(){
        var timeline = []
        range(0, this.n_conditions, 1).map(i => {
            var p = i / this.base

            timeline.push(pressSpacebar(`
            <div>
                <h1>Practice</h1>
                The next <b>practice game</b> is in <b>Condition ${i+1}:</b><br/>
                there is a <b>${i} in ${this.base} (${p * 100}%)</b>
                ${this.interp}<br/><br/>
                Press the <b>spacebar</b> to continue
                </p>
            </div>
            `))

            timeline.push(game_nodes(this.create_game(p, "practice"), 
            `<p>Condition ${i + 1}: ${i} in ${this.base} (${i / this.base * 100}%) ${this.interp}</p>`))
        })
        return {timeline: timeline}
    }

    /**
     * Creates all of the game blocks
     * @param {Number} games_per_block 
     */
    game_blocks(games_per_block){
        var timeline = []

        // When the intro is skipped (demo ?skip=1) there was no tutorial to
        // finish, so give a short orientation instead.
        timeline.push(pressSpacebar(config.skip_intro ? `
        <h1>Ready?</h1>
        You are an explorer collecting treasure. Starting at the top of the map,
        you move down one row at a time, pressing the <b>F key to move left</b>
        and the <b>J key to move right</b>, and collect the points in the
        chests you land on.<br/>
        <b>Your goal is to get the most points possible.</b>
        <br/><br/>
        You will play ${this.n_conditions * 2} sets of ${games_per_block} games each.
        <br/><br/>
        Press the <b>spacebar</b> to continue.
        ` : `
        <h1>Ready?</h1>
        You have now finished the tutorial and will move on
        to the real game. <br/>
        You will play ${this.n_conditions * 2} sets of ${games_per_block} games each.
        <br/><br/>
        Press the <b>spacebar</b> to continue.
        `))

        var conditions = range(0, this.n_conditions, 1)
        conditions = shuffle(conditions)
        var reversed_conditions = deepcopy(conditions).reverse()

        conditions.map((i) =>{
            var p = i / this.base

            timeline.push(pressKey(`
            <div>
                <h1>Play</h1>
                <h3>
                If you wish, you may take a break before moving on.<br/>
                The next <b>${games_per_block} games</b> are<br/><br/>
                </h3>
                <h2>
                <b>Condition ${i+1}:</b><br/>
                there is a <b>${i} in ${this.base} (${p * 100}%)</b>
                ${this.interp}<br/><br/>
                </h2>

                <h3>Press the "<b>${i+1}</b>" Key to continue</h3>
                </p>
            </div>
            `, `${i+1}`))

            range(0, games_per_block, 1).map(() => {
                timeline.push(game_nodes(this.create_game(p), 
                `<p>Condition ${i + 1}: ${i} in ${this.base} (${i / this.base * 100}%) ${this.interp}</p>`))  
            })
        })

        reversed_conditions.map((i) =>{
            var p = i / this.base

            timeline.push(pressKey(`
            <div>
                <h1>Play</h1>
                <h3>
                If you wish, you may take a break before moving on.<br/>
                The next <b>${games_per_block} games</b> are<br/><br/>
                </h3>
                <h2>
                <b>Condition ${i+1}:</b><br/>
                there is a <b>${i} in ${this.base} (${p * 100}%)</b>
                ${this.interp}<br/><br/>
                </h2>

                <h3>Press the "<b>${i+1}</b>" Key to continue</h3>
                </p>
            </div>
            `, `${i+1}`))

            range(0, games_per_block, 1).map(() => {
                timeline.push(game_nodes(this.create_game(p), 
                `<p>Condition ${i + 1}: ${i} in ${this.base} (${i / this.base * 100}%) ${this.interp}</p>`))  
            })
        })
        
        // Attention checks warn about failing the study and not being paid,
        // which is meaningless (and alarming) for a demo visitor.
        if (!config.demo){
            timeline.splice(1 + (1 + games_per_block) * 2, 0, this.attention_check())
            timeline.splice(2 + (1 + games_per_block) * 5, 0, this.attention_check())
            timeline.splice(3 + (1 + games_per_block) * 8, 0, this.attention_check())
        }

        return {timeline: timeline}
    }
}

export class Reliability extends StochasticityType{
    constructor(n_conditions){
        super("reliability", n_conditions, n_conditions - 1, "chance that a given treasure chest is a mystery chest")
    }

    instructions(){
        var instruction_stack = []
        instruction_stack.push(
            instruction_content(
                `
                <h2>Now we will add a little more to the task.
                </h2>
                <br/><br/>
                Press <b>Next</b> to continue.
                <br/><br/>
                `
            )
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Treasure types
                </h2>
                In our map, we have two types of treasure chests: 
                <b>normal</b> treasure chests and <b>mystery</b> treasure
                chests. 
                <br/><br/>
                <b>You will not know in advance whether a treasure chest
                is a normal or mystery chest.</b> You will only find out once you choose it. <br/><br/>
                `, 
                "closed.png"
            ), 
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Treasure types
                </h2>
                <b>Normal</b> treasure chests contain
                the same number of points as written on the front.
                <br/><br/>
                <b>Mystery</b> treasure chests contain a <b>random number</b> of points between <b>1 and 9</b>, 
                regardless of what's written on the front.
                <br/><br/>
                If the treasure chest you choose is a mystery chest, it will shake, turn blue, and then open
                to reveal the points inside.
                <br/><br/>
                `, 
                "reliability.png"
            ), 
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Treasure types
                </h2>
                <b>There is no pattern to whether a treasure chest is a normal or mystery chest.</b><br/><br/>
                Whether a treasure chest is a normal chest or a mystery chest <b>does not depend</b> on the value
                of the treasure chest, the value of other treasure chests, the location of the treasure chest, or
                your past history of moves in any way.
                `, 
                "reliability.png"
            ), 
        )

        return instruction_stack
    }

    comprehension_check(){
        return [
            {
                text: `<h3>In condition 3 there is a 50% chance
                that a given chest will be a mystery chest. If there 
                are 10 treasure chests in total ... 
                </h3>
                 
                `, 
                options: [
                    "Exactly 5 chests will be mystery chests",
                    "Less than 5 chests will be mystery chests",
                    "More than 5 chests will be mystery chests", 
                    "It is impossible to know"
                ], 
                desc: `In this case it is impossible to know exactly how many mystery chests there will be.
                We may expect that around half of the chests will be mystery chests, but we cannot be certain 
                that we get exactly half due to randomness. The number of mystery chests may be more, equal, 
                or less than half of the chests.
                `
                ,
                answer: 3
            },
            {
                text: `<h3>In condition 1 there is a 0% chance that a 
                given chest will be a mystery chest. If there are 10 treasure chests
                in total ...
                </h3>`
                , 
                options: [
                    "10 chests will be mystery chests",
                    "0 chests will be mystery chests",
                    "5 chests will be mystery chests",
                    "It is impossible to know"
                ], 
                desc: `In this case we know that there is a 0% chance that any given treasure chest will be a mystery chest. So none of the chests
                will be mystery chests.`
                ,
                answer: 1
            },
            {
                text: `<h3>A treasure chest is more likely to be a mystery chest if ...
                </h3>`
                , 
                options: [
                    "It has a large number written on the front",
                    "Its neighbors are also mystery treasure chests",
                    "The last treasure chest you visited wasn't a mystery treasure chest",
                    "None of the above"
                ], 
                desc: `Whether a treasure chest is normal or mystery is completely random. It is not related to
                the treasure chest's location, value, or your previous moves.
                `,
                answer: 3
            },
            {
                text: `<h3> You see an unopened treasure chest with a 5 written on the front.
                When you visit it, you discover it is a mystery chest. The number of points you get...
                </h3>`
                , 
                options: [
                    "must be equal to 5",
                    "must not be equal to 5",
                    "Can be any number between 1 and 9",
                    "None of the above"
                ], 
                desc: `The value of the mystery treasure chest is a random number between 1 and 9. 
                It could be greater than, less than, or equal to the number written on the front.
                `,
                answer: 2
            },
        ]
    }

    create_game(p, prefix){
        return new CustomGame(prefix, 8, [p, 0.0, 0.0], false)
    }
}

export class Controllability extends StochasticityType{
    constructor(n_conditions){
        super("transition_noise", n_conditions, (n_conditions - 1)*2, "chance that a given move will be flipped")
    }

    instructions(){
        var instruction_stack = []
        instruction_stack.push(
            instruction_content(
                `
                <h2>Now we will add a little more to the task.
                </h2>
                <br/><br/>
                Press <b>Next</b> to continue.
                <br/><br/>
                `
            )
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Stormy seas
                </h2>
                Sometimes the sea can be unpredictable and take
                you in directions you do not expect. Sometimes your
                move may be <b>flipped</b>.<br/><br/>
                For example, if you press F, instead of moving left
                as you normally would, your move will be flipped and you will 
                go right instead. <br/><br/>
                You will not know in advance which moves will be flipped.
                `, 
                "transition_flipped.png"
            ), 
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Flipped moves
                </h2>
                <b> There is no pattern to which moves are flipped. </b>
                Whether a move is flipped <b>does not depend</b> on your past actions, the value of the
                treasure chests, or whether your moves were flipped before.
                `,
                "transition_flipped.png"
            ), 
        )
        
        return instruction_stack
    }

    comprehension_check(){
        return [
            {
                text: `<h3>In a flipped move, if you press the F (left) key ... 
                </h3>
                 
                `, 
                options: [
                    "You will move left",
                    "You will move right",
                    "You will move up", 
                    "It is impossible to know"
                ], 
                desc: `In a flipped move, you will move right if you press the F key.
                `
                ,
                answer: 1
            },
            {
                text: `<h3>In condition 3 there is a 25% chance
                that a given move will be flipped. For every 4 moves... 
                </h3>
                 
                `, 
                options: [
                    "Exactly 1 move will be flipped",
                    "Less than 1 move will be flipped",
                    "More than 1 move will be flipped", 
                    "It is impossible to know"
                ], 
                desc: ` It is impossible to know.
                We may expect that around 1 move will be flipped, but we cannot be certain. There may
                be more, less, or exactly 1 flipped move.
                `
                ,
                answer: 3
            },
            {
                text: `<h3>In condition 1 there is a 0% chance
                that a given move will be flipped. For every 4 moves... 
                </h3>
                 
                `, 
                options: [
                    "No moves will be flipped",
                    "All moves will be flipped",
                    "Half of the moves will be flipped", 
                    "It is impossible to know"
                ], 
                desc: `No moves will be flipped. We know that there is a 0% chance
                that any move will be flipped, so we can be certain that none of the 4 moves will 
                be flipped.
                `
                ,
                answer: 0
            },
            {
                text: `<h3>A move is more likely to be flipped if... 
                </h3>
                 
                `, 
                options: [
                    "The previous move was flipped",
                    "The treasure chests in the future have high values",
                    "You move left", 
                    "None of the above"
                ], 
                desc: `Whether a move is flipped or not is completely random. It does not depend on the value of the
                treasure chests, whether a previous move was flipped, or your past actions.
                `
                ,
                answer: 3
            },
        ]

    }
    
    create_game(p, prefix){
        return new CustomGame(prefix, 8, [0.0, 0.0, p], false)
    }
}

export class Volatility extends StochasticityType{
    constructor(n_conditions){
        super("volatility", n_conditions, n_conditions - 1, "chance that a given treasure chest in the earthquake zone will shake")
    }

    instructions(){
        var instruction_stack = []

        instruction_stack.push(
            instruction_content(
                `
                <h2>Now we will add a little more to the task.
                </h2>
                <br/><br/>
                Press <b>Next</b> to continue.
                <br/><br/>
                `
            )
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Earthquake
                </h2>
                Every move, an earthquake will 
                randomly shake some of the treasure chests 
                in the rows below you. We will call this
                region the <b>earthquake zone.</b><br/><br/>
                
                As you move down, 
                the earthquake zone will also move down.<br/><br/>
    
                Please study the diagram at the right.
                `, 
                "earthquake_zone.png"
            ), 
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Earthquake
                </h2>
                Treasure chests in the earthquake zone can turn blue and shake.<br/><br/>
                The shaking treasure chests will be <b>randomly assigned new values between 1 and 9</b>.
                <br/><br/>
    
                You will not know in advance which treasure chests will 
                shake. Treasure chests that shake during a given move may or may not
                shake in the next move. It is impossible to know for sure which treasure
                chests will shake.
                `,
                "volatility.png"
            ), 
        )
        instruction_stack.push(
            instruction_content(
                `
                <h2>Earthquake
                </h2>
                <b>There is no pattern to whether a treasure chest in the earthquake zone will shake.</b><br/><br/>
                Whether a treasure chest shakes <b>does not depend</b> on the value of the chest, 
                the location within the earthquake zone, whether it has shaken in the past, or your past history of moves
                in any way.
                `, 
                "volatility.png"
            ), 
        )
        
        return instruction_stack
    }

    comprehension_check(){
        return [
            {
                text: `<h3>In condition 3 there is a 50% chance
                that a given chest in the earthquake zone will shake. If there 
                are 10 treasure chests in the earthquake zone ... 
                </h3>
                 
                `, 
                options: [
                    "Exactly 5 chests will shake",
                    "Less than 5 chests will shake",
                    "More than 5 chests will shake", 
                    "It is impossible to know"
                ], 
                desc: `In this case it is impossible to know exactly how many chests will shake.
                We may expect that around half of the chests will shake, but we cannot be certain 
                that we get exactly half due to randomness. The number of shaking chests may be more, equal, 
                or less than half of the chests.
                `
                ,
                answer: 3
            },
            {
                text: `<h3>In condition 1 there is a 0% chance that a 
                given chest in the earthquake zone will shake. If there are 10 treasure chests
                in the earthquake zone ...
                </h3>`
                , 
                options: [
                    "Exactly 10 chests will shake",
                    "Exactly 0 treasure chests will shake",
                    "Around 5 treasure chests will shake",
                    "It is impossible to know"
                ], 
                desc: "In this case we know that each chest has a 0% chance of shaking. This means that none of the treasure chests will shake.",
                answer: 1
            },
            {
                text: `<h3>On a given move, a treasure chest is more likely shake if ...
                </h3>`
                , 
                options: [
                    "It has a large number written on the front",
                    "Its neighbors also shake",
                    "It did not shake in the previous move",
                    "None of the above"
                ], 
                desc: `Whether a treasure chest will shake is completely random. It is not related to
                the treasure chest's location within the earthquake zone, value, or whether it shook in the past.
                `,
                answer: 3
            },
            {
                text: `<h3>One of the treasure chests in the earthquake zone is shaking. 
                If it had a 3 written on the front before it started shaking, the value of the treasure chest
                after shaking is ...
                </h3>`
                , 
                options: [
                    "Not equal to 3",
                    "Smaller than 3",
                    "A random value between 1 and 9",
                    "None of the above"
                ], 
                desc: `The value of the treasure chest after shaking is a random number between 1 and 9. 
                It could be greater than, less than, or equal to 3.
                `,
                answer: 2
            },
            {
                text: `<div class="img-container"><img src='images/earthquake_zone_comprehension1.png' height=300/></div><br/><br/>
                       <h3>Which treasure chests are in the earthquake zone?</h3>`
                , 
                options: [
                    "A only",
                    "B and C",
                    "B, C, and D",
                    "D only",
                    "All of the above"
                ], 
                desc: "B, C, and D are all in the earthquake zone",
                answer: 2
            },
        ]
    }
    
    create_game(p, prefix){
        return new CustomGame(prefix, 8, [0.0, p, 0.0], true)
    }

    create_attention_node(){
        return attention_nodes(true)
    }
}
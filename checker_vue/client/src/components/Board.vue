<template>
  <div>
    <div id="game-view-info" >
        <div v-if="gamestate == 1" style="color:mediumseagreen;">YOUR TURN !!!</div>
        <div v-if="gamestate == 3" style="color:green;">GAME END YOU WIN !!!</div>
        <div v-if="gamestate == -1" style="color:salmon;">AI TURN PLEASE WAIT !!!</div>
        <div v-if="gamestate == -3" style="color:crimson;">GAME END AI WIN !!!</div>
    </div>
    <div id="game-view-squres">
      <div 
        v-for="item in StatusBoard" :key="item.index"
        class="game-view-squre"
        v-on:click="ClickToMove(item.index)"
        v-bind:class="{'bluebloack': (item.status == 2),
                      'redbloack': ((Math.floor(item.index/8)%2) && !((item.index%2) == 0)) 
                                  || (!(Math.floor(item.index/8)%2)&& ((item.index%2)== 0))}">
        <img v-if="item.data == 1" src="white_sprite.png" id="image-style">
        <img v-if="item.data == 3" src="white_king_sprite.png" id="image-style">
        <img v-if="item.data == -1" src="black_sprite.png" id="image-style">
        <img v-if="item.data == -3" src="black_king_sprite.png" id="image-style">
      </div>
    </div>
  <div id="game-view-panel">
      <button 
      v-on:click="resetGame()"
      class="button button2 reset">RESET</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Board',
  props: {
    msg: String
  },
  data() {
    return {
      board: [],
      StatusBoard: [],
      startpos: null,
      gamestate: 1,

      aicancelkey: null //for cancel long request and reset without delayed respond
    };
  },
  methods: {
    getBoard() {
      this.gamestate = 1
      const path = 'http://localhost:5000/newgame';
      axios.get(path)
        .then((res) => {
          this.board = res.data.board;
          this.resetStatus(this.board);
          this.getMove();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },


    getMove(){
      const path = 'http://localhost:5000/getpossiblemove';
      axios.post(path, this.board)
        .then((res) => {
          var moves = res.data.possible_moves;
          for(var x in moves){
            this.StatusBoard[x].possiblemove = moves[x]
          }
        })
        .catch((error) => {
          console.log(error);
        });
    },

    //check status when click hilighted block => make move
    //else => get new hilighted block
    ClickToMove(idx){
      if(this.gamestate == 1){
        var sb = this.StatusBoard
        if(sb[idx].status == 2){
          // console.log('Start = ' + String(this.startpos));
          // console.log('End = ' + String(idx));
          this.MakeMove(idx);
        }else{
          for(var x in sb){
            sb[x].status = 0
          }
          sb[idx].status = 1
          this.startpos = idx
          for(var y of sb[idx].possiblemove){
            sb[y].status = 2
          }
        }
      }
    },

    MakeMove(endpos){
      const path = 'http://localhost:5000/makemove';
      var payload = {}

      payload['board'] = this.board
      payload['start_pos'] = this.startpos
      payload['end_pos'] = endpos
      axios.post(path, payload)
        .then((res) => {
          this.board = res.data.board;
          var result = res.data.result;
          this.resetStatus(this.board);
          this.GetGameEnd(result)
          if(this.gamestate == -1){
            this.GetAiMove();
          }
        })
        .catch((error) => {
          console.log(error);
      });
    },

    GetAiMove(){
      const path = 'http://localhost:5000/aimove';
      this.aicancelkey = axios.CancelToken.source();
      axios.post(path, this.board, { cancelToken: this.aicancelkey.token })
        .then((res) => {
          this.board = res.data.board;
          var result = res.data.result;
          this.resetStatus(this.board);
          this.GetGameEnd(result)
          if(this.gamestate == 1){
            this.getMove();
          }
        })
        .catch((error) => {
          console.log(error);
      });
    },

    GetGameEnd(result){
      if(result == 0){
        this.gamestate = -this.gamestate
      }else{
        this.gamestate = 3*result
      }
    },


    resetStatus(board){
      var i;
      var satusData = []
      for (i = 0; i < board.length; i++) {
        satusData.push({'index':i, 'data':board[i], 'status':0, 'possiblemove':[]})
      }
      //console.log(satusData)
      this.StatusBoard = satusData
    },

    resetGame(){
      try {
            this.aicancelkey.cancel('aimove request is canceled');
          }
      catch(error) {
        console.error(error);
      }
      this.getBoard();
      
    }


  },
  created() {
    this.getBoard();
  }
}




</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style >
#game-view-squres{
  height: 750px;
  display: flex;
  flex-wrap: wrap;
  box-sizing: border-box;
}

#image-style{
  width: 75%;
  height: 75%;
  justify-content: center;
  align-items: center;
}

.game-view-squre{
  width: 12.5%;
  height: 12.5%;
  background-color: #ffe57c;

  display: flex;
  justify-content: center;
  align-items: center;
  box-sizing: border-box;

  font-family: cursive;
  font-size: 75px;
  text-transform:  uppercase;

  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
}


.game-view-squre.redbloack{background-color: #ff6961;}
.game-view-squre.bluebloack{background-color: #82eefd;}

.game-view-squre:hover{background-color: #00ffcd;}


.whitedot {
  height: 75%;
  width: 75%;
  background-color: #ffffff;
  border-radius: 50%;
  display: inline-block;
}

.blackdot {
  height: 75%;
  width: 75%;
  background-color: #000000;
  border-radius: 50%;
  display: inline-block;
}


#game-view-info{    
    padding: 10px;
    font-family:sans-serif;
    font-size: 40px;
    font-weight: bold;

    margin: auto;
    text-align: center;
    justify-content: center;
    background-color: #eee;

}

#game-view-panel{
    padding: 15px;
    
    font-family:sans-serif;
    font-size: 20px;
    font-weight: bold;

    text-align: center;
    background-color: #eee;
}

.button {
  background-color: 	deepskyblue; /* Green */
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 20px;
  margin: 4px 10px;
  cursor: pointer;
  -webkit-transition-duration: 0.4s; /* Safari */
  transition-duration: 0.4s;
}

.button2:hover {
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);
}

.reset{
  background-color: red; /* Green */
}



</style>

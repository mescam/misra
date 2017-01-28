import sys
import collections
from enum import Enum

import colorama
from mpi4py import MPI


comm = MPI.COMM_WORLD


class NodeState(Enum):
    NO_TOKEN = 0
    TOKEN_PING = 1
    TOKEN_PONG = 2
    BOTH_TOKENS = 3


class MessageTag(Enum):
    TOKEN = 0


class TokenType(Enum):
    PING = 0
    PONG = 1


class Logger:

    colors = {
        'info': colorama.Fore.GREEN,
        'warning': colorama.Fore.YELLOW,
        'critical': colorama.Fore.RED
    }

    def __init__(self):
        self.rank = comm.rank

    def _message(self, color, message):
        print(colorama.Fore.CYAN + ("[%02d]\t" % self.rank) +
              color + message, file=sys.stderr)

    def info(self, message):
        self._message(Logger.colors['info'], message)

    def warning(self, message):
        self._message(Logger.colors['warning'], message)

    def critical(self, message):
        self._message(Logger.colors['critical'], message)


class Message:
    def __init__(self, message, tag):
        self.message = message
        self.tag = tag.value

    def send(self, recipients):
        if isinstance(recipients, collections.Iterable):
            for i in recipients:
                comm.send(self.message, i, self.tag)
        else:
            comm.send(self.message, recipients, self.tag)

    def broadcast(self):
        for node in range(comm.size):
            if node != comm.rank:
                comm.send(self.message, node, self.tag)


class Token:

    def __init__(self, token_type, value):
        if not instance(token_type, TokenType):
            raise RuntimeError("Token should be TokenType instance")
        self.type = token_type
        self.value = value

    def __str__(self):
        return "TOKEN %s(%d)" % (self.type, self.value)

    def send(self, recipient):
        Logger().info("Sending %s to node %d" % (self, recipient))
        Message(self, MessageTag.TOKEN).send(recipient)


class Node:

    def __init__(self, comm):
        self.rank = comm.rank
        self.next_node = (comm.rank + 1) % comm.size
        self.m = 0
        self.state = NodeState.NO_TOKEN
        self.log = Logger()
        self.log.info("initialized node")

    def run(self):
        self.log.info("starting node")
        if self.rank == 0:
            self.log.warning("I am the chosen one, generating first tokens")
            Token(TokenType.PING, 1).send(self.next_node)
            Token(TokenType.PONG, 1).send(self.next_node)

        self.listen()

    def listen(self):
        token = comm.recv()
        self.log.info("received token %s" % token)
        self.become(token)

    def become(self, token):
        # let the ifs begin
        if self.token.value == self.m:
            if self.m > 0:
                self.log.warning('PONG TOKEN LOST')
                self.regenerate_token(TokenType.PONG)
            else:
                self.log.warning('PING TOKEN LOST')
                self.regenerate_token(TokenType.PONG)

        if token == TokenType.PING:
            if self.state == NodeState.NO_TOKEN:
                self.state = NodeState.TOKEN_PING

            elif self.state == NodeState.TOKEN_PONG:
                self.state = NodeState.BOTH_TOKENS

            elif self.state in [NodeState.TOKEN_PING, NodeState.BOTH_TOKENS]:
                pass

    def incarnate_tokens(self, x):
        return Token(TokenType.PING, abs(x)),\
            Token(TokenType.PONG, - abs(x + 1))

    def regenerate_token(self, x):
        pass


if __name__ == '__main__':
    Node(comm).run()

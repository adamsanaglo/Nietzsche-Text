const Discord = require('discord.js');
const client = new Discord.Client();
const NLP = require('natural');
const classifier = new NLP.LogisticRegressionClassifier();




client.once('ready', () => {
	console.log('Ready!');
});
const prefix = "!";
client.on('message', message => {
	console.log(message.content);

	const avatarEmbed = new Discord.MessageEmbed()
	.setColor('#0099ff')
	.setTitle('')
	.setURL('https://discord.js.org/')
	.setAuthor('Some name', 'https://i.imgur.com/wSTFkRM.png', 'https://discord.js.org')
	.setDescription('Some description here')
	.setImage('https://i.imgur.com/wSTFkRM.png')
	.setTimestamp();



	if (!message.content.startsWith(prefix) || message.author.bot) return;

    const args = message.content.slice(prefix.length).split(/ +/);
    const command = args.shift().toLowerCase();

    if (command === 'args-info') {
	if (!args.length) {
		return message.channel.send(`You didn't provide any arguments, ${message.author}!`);
	}

	message.channel.send(`Command name: ${command}\nArguments: ${args} `);
}

	 if (message.content === "!hello") {
    message.channel.send("Hi, please send \"!help\" to know what I can help you with")
    }


    if (command === 'avatar') {
	if (!message.mentions.users.size) {
		avatarlink = message.author.displayAvatarURL({ format: "png", dynamic: true });
		avatarEmbed.setImage(avatarlink);
        message.channel.send(avatarEmbed);

	}

	const avatarList = message.mentions.users.map(user => {
		avatarlink = user.displayAvatarURL({ format: "png", dynamic: true });
		avatarEmbed.setImage(avatarlink);
		message.channel.send(avatarEmbed);

	});

	message.channel.send(avatarList);
}
    }
 );


client.login('NzA5MDAzMzQ5MjgxOTk2ODIx.XtqF_g.FA7GJu2gfKNC116A_fr3MX5mEKA');















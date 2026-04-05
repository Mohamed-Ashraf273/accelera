const isUrl = require('is-url');

const isUrlValidation = (value) => {
    return isUrl(value);
}
module.exports = {  isUrlValidation }